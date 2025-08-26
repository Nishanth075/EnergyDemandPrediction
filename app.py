import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import tempfile
from scipy import stats
import holidays

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Supported file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_datetime_column(df):
    """Automatically detect datetime column in the dataframe"""
    datetime_cols = []
    for col in df.columns:
        # Try to convert to datetime
        try:
            pd.to_datetime(df[col].head(1000))
            datetime_cols.append(col)
        except:
            continue
    
    if not datetime_cols:
        return None
    
    # Prefer columns with 'date', 'time' in name
    preferred_cols = [col for col in datetime_cols if 'date' in col.lower() or 'time' in col.lower()]
    return preferred_cols[0] if preferred_cols else datetime_cols[0]

def detect_demand_column(df):
    """Automatically detect energy demand column in the dataframe"""
    demand_keywords = ['mw', 'load', 'consumption', 'demand', 'usage', 'energy']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in demand_keywords):
            return col
    
    # If no obvious match, return first numeric column that's not datetime
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    datetime_col = detect_datetime_column(df)
    if datetime_col in numeric_cols:
        numeric_cols = numeric_cols.drop(datetime_col)
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def preprocess_data(df, datetime_col, demand_col):
    """Preprocess the input data"""
    # Convert to datetime and set as index
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values - forward fill for time series
    df[demand_col] = df[demand_col].fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers using IQR
    Q1 = df[demand_col].quantile(0.25)
    Q3 = df[demand_col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[demand_col] >= (Q1 - 1.5 * IQR)) & (df[demand_col] <= (Q3 + 1.5 * IQR))]
    
    return df

def create_features(df, demand_col):
    """Create time-based features"""
    df = df.copy()
    
    # Extract time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    # Weekend flag
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Season (1:Winter, 2:Spring, 3:Summer, 4:Fall)
    df['season'] = df['month'].apply(lambda month: (month % 12 + 3)//3)
    
    # Holiday flag (US holidays)
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.to_series().apply(lambda x: x in us_holidays).astype(int)
    
    # Lag features
    df['demand_lag1'] = df[demand_col].shift(1)
    df['demand_lag2'] = df[demand_col].shift(2)
    df['demand_lag3'] = df[demand_col].shift(3)
    
    # Rolling features
    df['demand_rolling_mean_7'] = df[demand_col].rolling(window=7).mean()
    df['demand_rolling_std_7'] = df[demand_col].rolling(window=7).std()
    
    # Drop rows with NaN values from lag features
    df.dropna(inplace=True)
    
    return df

def detect_frequency(df):
    """Detect the frequency of the time series data"""
    if len(df) < 2:
        return None
    
    time_diff = df.index.to_series().diff().dropna()
    mode_diff = time_diff.mode()[0]
    
    if mode_diff <= pd.Timedelta('1 hour'):
        return 'hourly'
    elif mode_diff <= pd.Timedelta('1 day'):
        return 'daily'
    elif mode_diff <= pd.Timedelta('7 days'):
        return 'weekly'
    elif mode_diff <= pd.Timedelta('30 days'):
        return 'monthly'
    else:
        return 'unknown'

def train_models(X_train, y_train):
    """Train multiple models and return the best one"""
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    model_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        model_metrics[name] = {'RMSE': rmse, 'MAE': mae}
        
        if rmse < best_score:
            best_score = rmse
            best_model = model
    
    return best_model, model_metrics

def predict_future(model, last_data_point, demand_col, frequency, periods=30):
    """Generate future predictions"""
    future_dates = pd.date_range(start=last_data_point, periods=periods+1, freq=frequency[0])[1:]
    future_df = pd.DataFrame(index=future_dates)
    
    # Create features for future dates
    future_df = create_features(future_df, demand_col)
    
    # We need to handle lag features which would require previous actual values
    # For simplicity, we'll use the predicted values recursively
    # In a production environment, you'd want a more sophisticated approach
    
    # Predict
    X_future = future_df.drop(columns=[c for c in future_df.columns if c.startswith('demand_')], errors='ignore')
    future_predictions = model.predict(X_future)
    
    return future_df.index, future_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the CSV file
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                return render_template('index.html', error=f"Error reading file: {str(e)}")
            
            # Detect datetime and demand columns
            datetime_col = detect_datetime_column(df)
            demand_col = detect_demand_column(df)
            
            if not datetime_col or not demand_col:
                return render_template('index.html', error="Could not detect datetime or demand columns")
            
            # Preprocess data
            df = preprocess_data(df, datetime_col, demand_col)
            
            # Create features
            df = create_features(df, demand_col)
            
            # Detect frequency
            frequency = detect_frequency(df)
            if not frequency:
                return render_template('index.html', error="Could not detect data frequency")
            
            # Prepare features and target
            X = df.drop(columns=[demand_col])
            y = df[demand_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            best_model, model_metrics = train_models(X_train_scaled, y_train)
            
            # Save model and data for later use
            model_filename = os.path.join('models', f'model_{filename}.joblib')
            os.makedirs('models', exist_ok=True)
            joblib.dump(best_model, model_filename)
            
            # Save processed data
            processed_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            df.to_csv(processed_data_path)
            
            # Prepare data for visualization
            plot_data = {
                'dates': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'demand': df[demand_col].tolist(),
                'demand_col': demand_col
            }
            
            return render_template('results.html', 
                                filename=filename,
                                datetime_col=datetime_col,
                                demand_col=demand_col,
                                frequency=frequency,
                                model_metrics=model_metrics,
                                plot_data=json.dumps(plot_data))
            
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    if request.method == 'POST':
        filename = request.form.get('filename')
        demand_col = request.form.get('demand_col')
        frequency = request.form.get('frequency')
        prediction_periods = int(request.form.get('prediction_periods', 30))
        
        # Load processed data
        processed_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        
        # Load model
        model_filename = os.path.join('models', f'model_{filename}.joblib')
        model = joblib.load(model_filename)
        
        # Prepare features for prediction
        X = df.drop(columns=[demand_col])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Make predictions on historical data
        historical_predictions = model.predict(X_scaled)
        
        # Predict future
        last_date = df.index[-1]
        future_dates, future_predictions = predict_future(model, last_date, demand_col, frequency, prediction_periods)
        
        # Create Plotly figure
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Historical actual
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[demand_col],
                name='Actual Demand',
                line=dict(color='blue'),
                mode='lines'
            )
        )
        
        # Historical predicted
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=historical_predictions,
                name='Model Fit',
                line=dict(color='green', dash='dot'),
                mode='lines'
            )
        )
        
        # Future predicted
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                name='Future Prediction',
                line=dict(color='red'),
                mode='lines'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Energy Demand Prediction',
            xaxis_title='Date',
            yaxis_title='Demand',
            hovermode='x unified',
            showlegend=True
        )
        
        # Convert figure to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('visualize.html', 
                             graphJSON=graphJSON,
                             filename=filename,
                             demand_col=demand_col)
    
    return redirect(url_for('index'))

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/download_processed/<filename>', methods=['GET'])
def download_processed(filename):
    processed_filename = f'processed_{filename}'
    return send_from_directory(app.config['UPLOAD_FOLDER'], processed_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)