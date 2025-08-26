// scripts.js
// Add your JavaScript code here
document.addEventListener('DOMContentLoaded', function() {
    // Add loading spinner during file upload and processing
    const fileUploadForm = document.getElementById('fileUploadForm');
    if (fileUploadForm) {
        fileUploadForm.addEventListener('submit', function() {
            const spinnerContainer = document.createElement('div');
            spinnerContainer.className = 'spinner-container';
            spinnerContainer.innerHTML = `
                <div class="spinner-border text-primary spinner" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="ms-3 text-white">Processing your data...</div>
            `;
            document.body.appendChild(spinnerContainer);
            spinnerContainer.style.display = 'flex';
        });
    }
    
    // Drag and drop functionality
    const dropZone = document.querySelector('.dropzone');
    const fileInput = document.getElementById('file');
    
    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.classList.add('bg-primary', 'bg-opacity-10');
        }
        
        function unhighlight() {
            dropZone.classList.remove('bg-primary', 'bg-opacity-10');
        }
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                fileInput.files = files;
            }
        }
    }
});