// ====================================
// BRISC 2025 - Segmentation Module
// ====================================

let selectedFile = null;

// DOM Elements
const uploadZone = document.querySelector('.upload-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.querySelector('.upload-placeholder');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const btnRemove = document.getElementById('remove-btn');
const btnSegment = document.getElementById('segment-btn');
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
        btnSegment.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove image
btnRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    btnSegment.disabled = true;
    
    // Reset results
    resultsContent.innerHTML = `
        <div class="empty-state">
            <i class="fas fa-image empty-icon"></i>
            <h4>No Segmentation Yet</h4>
            <p>Upload an image and click segment to see the tumor mask</p>
        </div>
    `;
});

// Segment button handler
btnSegment.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    // Show loading state
    resultsContent.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p style="color: var(--gray-600); font-weight: 600;">Segmenting tumor regions...</p>
        </div>
    `;
    
    btnSegment.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const response = await fetch('/api/segment', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
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
        btnSegment.disabled = false;
    }
});

// Display segmentation results
function displayResults(data) {
    const tumorPercentage = data.tumor_percentage ? data.tumor_percentage.toFixed(2) : 'N/A';
    const diceScore = data.dice_score ? (data.dice_score * 100).toFixed(2) : 'N/A';
    const iouScore = data.iou_score ? (data.iou_score * 100).toFixed(2) : 'N/A';
    
    resultsContent.innerHTML = `
        <div class="segmentation-display">
            <div class="segmentation-image">
                <h5><i class="fas fa-image"></i> Original Image</h5>
                <img src="${previewImage.src}" alt="Original">
            </div>
            <div class="segmentation-image">
                <h5><i class="fas fa-layer-group"></i> Tumor Segmentation</h5>
                <img src="data:image/png;base64,${data.mask_base64}" alt="Segmentation Mask">
            </div>
        </div>
        
        <div class="segmentation-stats">
            <h4><i class="fas fa-chart-line"></i> Segmentation Statistics</h4>
            <div class="stat-item">
                <span><i class="fas fa-percentage"></i> Tumor Coverage</span>
                <strong>${tumorPercentage}%</strong>
            </div>
            ${data.dice_score ? `
                <div class="stat-item">
                    <span><i class="fas fa-bullseye"></i> Dice Score</span>
                    <strong>${diceScore}%</strong>
                </div>
            ` : ''}
            ${data.iou_score ? `
                <div class="stat-item">
                    <span><i class="fas fa-vector-square"></i> IoU Score</span>
                    <strong>${iouScore}%</strong>
                </div>
            ` : ''}
            <div class="stat-item">
                <span><i class="fas fa-microchip"></i> Processing Time</span>
                <strong>${data.processing_time ? data.processing_time.toFixed(3) : 'N/A'}s</strong>
            </div>
        </div>
        
        <div class="info-box">
            <h4><i class="fas fa-info-circle"></i> Understanding Segmentation</h4>
            <ol>
                <li><strong>Tumor Coverage:</strong> Percentage of image area identified as tumor tissue</li>
                <li><strong>Dice Score:</strong> Measures overlap between predicted and ground truth (if available)</li>
                <li><strong>IoU Score:</strong> Intersection over Union metric for segmentation quality</li>
            </ol>
            <p style="margin-top: 1rem; color: var(--gray-600); font-size: 0.875rem;">
                <i class="fas fa-exclamation-circle"></i>
                <strong>Note:</strong> This is an AI-generated segmentation. Always consult medical professionals for accurate diagnosis.
            </p>
        </div>
    `;
}
