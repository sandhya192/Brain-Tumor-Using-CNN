// ====================================
// BRISC 2025 - Main JavaScript
// ====================================

// Model Status Checker
async function checkModelStatus() {
    const container = document.querySelector('.model-status-container');
    if (!container) return;

    try {
        const response = await fetch('/api/model-status');
        const data = await response.json();

        container.innerHTML = `
            <div class="status-grid">
                <div class="status-card ${data.classification_loaded ? 'status-active' : 'status-inactive'}">
                    <i class="fas fa-brain"></i>
                    <div>
                        <div class="status-label">Classification Model</div>
                        <div class="status-state">${data.classification_loaded ? 'Ready' : 'Not Loaded'}</div>
                    </div>
                </div>
                <div class="status-card ${data.segmentation_loaded ? 'status-active' : 'status-inactive'}">
                    <i class="fas fa-layer-group"></i>
                    <div>
                        <div class="status-label">Segmentation Model</div>
                        <div class="status-state">${data.segmentation_loaded ? 'Ready' : 'Not Loaded'}</div>
                    </div>
                </div>
                ${data.device ? `
                    <div class="status-card status-device">
                        <i class="fas fa-microchip"></i>
                        <div>
                            <div class="status-label">Compute Device</div>
                            <div class="status-state">${data.device}</div>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    } catch (error) {
        console.error('Error checking model status:', error);
        container.innerHTML = `
            <div class="status-card status-inactive">
                <i class="fas fa-exclamation-triangle"></i>
                <div>
                    <div class="status-label">Status</div>
                    <div class="status-state">Error loading status</div>
                </div>
            </div>
        `;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
});
