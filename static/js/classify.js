// ====================================
// BRISC 2025 - Classification Module
// ====================================

let selectedFile = null;

// DOM Elements
const uploadZone = document.querySelector('.upload-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.querySelector('.upload-placeholder');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const btnRemove = document.getElementById('remove-btn');
const btnClassify = document.getElementById('classify-btn');
const resultsContent = document.getElementById('results-content');

// File input change handler
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Click to upload
uploadPlaceholder.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop handlers
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        alert('Please drop a valid image file (JPEG, PNG)');
    }
});

// Handle file selection
function handleFileSelect(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        previewContainer.style.display = 'block';
        btnClassify.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove image
btnRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    btnClassify.disabled = true;
    
    // Reset results
    resultsContent.innerHTML = `
        <div class="empty-state">
            <i class="fas fa-chart-bar empty-icon"></i>
            <h4>No Results Yet</h4>
            <p>Upload an image and click classify to see predictions</p>
        </div>
    `;
});

// Classify button handler
btnClassify.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    console.log('🔬 Starting classification...');
    console.log('📁 File:', selectedFile.name, 'Size:', selectedFile.size, 'bytes');
    
    // Show loading state
    resultsContent.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p style="color: var(--gray-600); font-weight: 600;">Analyzing brain scan...</p>
        </div>
    `;
    
    btnClassify.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        console.log('📤 Sending request to /api/classify');
        const startTime = performance.now();
        
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });
        
        const endTime = performance.now();
        console.log(`⏱️ Request completed in ${(endTime - startTime).toFixed(0)}ms`);
        console.log('📥 Response status:', response.status);
        
        const data = await response.json();
        console.log('📊 Response data:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('✅ Classification successful!');
        console.log('Predicted:', data.predicted_class);
        console.log('Confidence:', (data.confidence * 100).toFixed(2) + '%');
        
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        resultsContent.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error: ${error.message}</p>
            </div>
        `;
    } finally {
        btnClassify.disabled = false;
    }
});

// Display classification results
function displayResults(data) {
    const predictedClass = data.predicted_class;
    const confidence = (data.confidence * 100).toFixed(2);
    
    // Create probability bars HTML
    const probabilityBars = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([className, prob]) => {
            const percentage = (prob * 100).toFixed(2);
            const isTop = className === predictedClass;
            
            return `
                <div class="probability-item">
                    <div class="probability-label">
                        <span style="text-transform: capitalize;">
                            ${isTop ? '<i class="fas fa-crown" style="color: var(--warning);"></i> ' : ''}
                            ${className.replace('_', ' ')}
                        </span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${percentage}%;">
                            ${percentage > 10 ? percentage + '%' : ''}
                        </div>
                    </div>
                </div>
            `;
        })
        .join('');
    
    resultsContent.innerHTML = `
        <div class="result-prediction">
            <h4><i class="fas fa-check-circle"></i> Prediction Result</h4>
            <div class="class-name">${predictedClass.replace('_', ' ')}</div>
            <div class="confidence">Confidence: ${confidence}%</div>
        </div>
        
        <div class="probability-bars">
            <h4>All Class Probabilities</h4>
            ${probabilityBars}
        </div>
        
        <div class="info-box">
            <h4><i class="fas fa-info-circle"></i> About This Prediction</h4>
            <p style="color: var(--gray-700); line-height: 1.7;">
                ${getClassDescription(predictedClass)}
            </p>
        </div>
    `;
}

// Get class descriptions
function getClassDescription(className) {
    const descriptions = {
        'glioma': 'Gliomas are tumors that arise from glial cells. They are the most common type of primary brain tumor and can vary in aggressiveness.',
        'meningioma': 'Meningiomas are typically benign tumors that arise from the meninges (protective membranes covering the brain and spinal cord).',
        'pituitary': 'Pituitary tumors develop in the pituitary gland at the base of the brain. They can affect hormone production and nearby structures.',
        'no_tumor': 'No tumor detected. The brain scan appears normal with no signs of abnormal growth or masses.'
    };
    
    return descriptions[className] || 'Classification complete. Please consult with a medical professional for diagnosis.';
}
