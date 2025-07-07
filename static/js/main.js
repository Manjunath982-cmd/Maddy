// Main JavaScript for FoodTracker Pro

// Global utilities
const FoodTracker = {
    // Initialize the application
    init: function() {
        this.setupGlobalEventListeners();
        this.initializeTooltips();
        this.setupAnimations();
        console.log('🍎 FoodTracker Pro initialized successfully!');
    },

    // Setup global event listeners
    setupGlobalEventListeners: function() {
        // Smooth scrolling for anchor links
        document.addEventListener('click', function(e) {
            if (e.target.matches('a[href^="#"]')) {
                e.preventDefault();
                const target = document.querySelector(e.target.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });

        // Auto-hide alerts after 5 seconds
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => {
            setTimeout(() => {
                if (alert.classList.contains('fade')) {
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 300);
                }
            }, 5000);
        });

        // Add loading state to form submissions
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', function(e) {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Processing...';
                    
                    // Re-enable after 10 seconds as fallback
                    setTimeout(() => {
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = submitBtn.getAttribute('data-original-text') || 'Submit';
                    }, 10000);
                }
            });
        });
    },

    // Initialize Bootstrap tooltips
    initializeTooltips: function() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },

    // Setup CSS animations
    setupAnimations: function() {
        // Intersection Observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Observe cards and sections for animation
        const animatedElements = document.querySelectorAll('.card, .hero-section, .stats-card');
        animatedElements.forEach(el => observer.observe(el));
    },

    // Utility functions
    utils: {
        // Format numbers with proper decimal places
        formatNumber: function(num, decimals = 1) {
            return Number(parseFloat(num).toFixed(decimals));
        },

        // Format calories display
        formatCalories: function(calories) {
            if (calories >= 1000) {
                return `${(calories / 1000).toFixed(1)}k`;
            }
            return Math.round(calories);
        },

        // Get nutrition color based on value
        getNutritionColor: function(nutrient, value) {
            const colors = {
                calories: value > 400 ? '#dc3545' : value > 200 ? '#ffc107' : '#28a745',
                protein: value > 20 ? '#28a745' : value > 10 ? '#ffc107' : '#dc3545',
                carbs: value > 50 ? '#17a2b8' : '#6c757d',
                fat: value > 20 ? '#dc3545' : value > 10 ? '#ffc107' : '#28a745'
            };
            return colors[nutrient] || '#6c757d';
        },

        // Show notification toast
        showToast: function(message, type = 'info') {
            const toastHtml = `
                <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">${message}</div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;
            
            // Create toast container if it doesn't exist
            let toastContainer = document.querySelector('.toast-container');
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
                document.body.appendChild(toastContainer);
            }
            
            // Add toast to container
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            const toast = toastContainer.lastElementChild;
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
            
            // Remove toast element after it's hidden
            toast.addEventListener('hidden.bs.toast', () => toast.remove());
        },

        // Copy text to clipboard
        copyToClipboard: function(text) {
            navigator.clipboard.writeText(text).then(() => {
                this.showToast('Copied to clipboard!', 'success');
            }).catch(() => {
                this.showToast('Failed to copy to clipboard', 'danger');
            });
        },

        // Download data as JSON
        downloadJSON: function(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    },

    // API helpers
    api: {
        // Generic API call wrapper
        call: async function(endpoint, options = {}) {
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json',
                },
            };

            const config = { ...defaultOptions, ...options };
            
            try {
                const response = await fetch(endpoint, config);
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'API call failed');
                }
                
                return data;
            } catch (error) {
                console.error('API Error:', error);
                FoodTracker.utils.showToast(error.message, 'danger');
                throw error;
            }
        },

        // Upload image for analysis
        uploadImage: async function(file, onProgress) {
            const formData = new FormData();
            formData.append('image', file);

            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid response format'));
                        }
                    } else {
                        reject(new Error('Upload failed'));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error'));
                });

                xhr.open('POST', '/upload');
                xhr.send(formData);
            });
        }
    },

    // Food analysis helpers
    food: {
        // Validate image file
        validateImage: function(file) {
            const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
            const maxSize = 16 * 1024 * 1024; // 16MB

            if (!validTypes.includes(file.type)) {
                return { valid: false, error: 'Please select a valid image file (JPEG, PNG, or WebP)' };
            }

            if (file.size > maxSize) {
                return { valid: false, error: 'Image size must be less than 16MB' };
            }

            return { valid: true };
        },

        // Calculate BMI
        calculateBMI: function(weight, height) {
            const heightInMeters = height / 100;
            const bmi = weight / (heightInMeters * heightInMeters);
            return Math.round(bmi * 10) / 10;
        },

        // Get BMI category
        getBMICategory: function(bmi) {
            if (bmi < 18.5) return { category: 'Underweight', color: '#17a2b8' };
            if (bmi < 25) return { category: 'Normal', color: '#28a745' };
            if (bmi < 30) return { category: 'Overweight', color: '#ffc107' };
            return { category: 'Obese', color: '#dc3545' };
        },

        // Calculate daily calorie needs
        calculateDailyCalories: function(age, weight, height, gender, activityLevel) {
            // Mifflin-St Jeor Equation
            let bmr;
            if (gender === 'male') {
                bmr = 10 * weight + 6.25 * height - 5 * age + 5;
            } else {
                bmr = 10 * weight + 6.25 * height - 5 * age - 161;
            }

            const activityMultipliers = {
                sedentary: 1.2,
                lightly_active: 1.375,
                moderately_active: 1.55,
                very_active: 1.725,
                extremely_active: 1.9
            };

            return Math.round(bmr * (activityMultipliers[activityLevel] || 1.55));
        }
    }
};

// Chart utilities
const ChartUtils = {
    // Default chart options
    getDefaultOptions: function() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            }
        };
    },

    // Create nutrition donut chart
    createNutritionChart: function(ctx, protein, carbs, fat) {
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Protein', 'Carbohydrates', 'Fat'],
                datasets: [{
                    data: [protein, carbs, fat],
                    backgroundColor: [
                        '#ff6384',
                        '#36a2eb',
                        '#ffcd56'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                ...this.getDefaultOptions(),
                cutout: '60%',
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    },

    // Create weight tracking chart
    createWeightChart: function(ctx, data) {
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [{
                    label: 'Weight (kg)',
                    data: data.map(d => d.weight),
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#28a745',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5
                }]
            },
            options: {
                ...this.getDefaultOptions(),
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    FoodTracker.init();
});

// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    FoodTracker.utils.showToast('An unexpected error occurred', 'danger');
});

// Export for global access
window.FoodTracker = FoodTracker;
window.ChartUtils = ChartUtils;