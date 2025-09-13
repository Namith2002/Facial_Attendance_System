// static/js/main.js

// Global variables
let currentTime = new Date();
let recognitionInterval;
let attendanceChart;
let videoFeedActive = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

function initializePage() {
    // Update current time every second
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
    
    // Initialize page-specific functionality
    const currentPage = getCurrentPage();
    
    switch(currentPage) {
        case 'login':
            initLoginPage();
            break;
        case 'dashboard':
            initDashboard();
            break;
        case 'register_face':
            initFaceRegistration();
            break;
        case 'attendance':
            initAttendancePage();
            break;
    }
}

function getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('login') || path === '/') return 'login';
    if (path.includes('dashboard')) return 'dashboard';
    if (path.includes('register_face')) return 'register_face';
    if (path.includes('attendance')) return 'attendance';
    return 'unknown';
}

function updateCurrentTime() {
    currentTime = new Date();
    const timeElement = document.getElementById('currentTime');
    if (timeElement) {
        timeElement.textContent = currentTime.toLocaleString();
    }
}

// Login Page Functions
function initLoginPage() {
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegistration);
    }
    
    // Add enter key support for forms
    document.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const activeForm = document.querySelector('form:not([style*="display: none"])');
            if (activeForm) {
                activeForm.dispatchEvent(new Event('submit'));
            }
        }
    });
}

function showLogin() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    
    if (loginForm && registerForm && toggleBtns.length >= 2) {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
        toggleBtns[0].classList.add('active');
        toggleBtns[1].classList.remove('active');
    }
}

function showRegister() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    
    if (loginForm && registerForm && toggleBtns.length >= 2) {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
        toggleBtns[0].classList.remove('active');
        toggleBtns[1].classList.add('active');
    }
}

async function handleRegistration(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Validate form data
    const username = formData.get('username');
    const password = formData.get('password');
    const email = formData.get('email');
    const fullName = formData.get('full_name');
    
    if (!username || !password || !email || !fullName) {
        showNotification('Please fill in all fields', 'error');
        return;
    }
    
    if (password.length < 6) {
        showNotification('Password must be at least 6 characters long', 'error');
        return;
    }
    
    if (!isValidEmail(email)) {
        showNotification('Please enter a valid email address', 'error');
        return;
    }
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Creating Account...';
    
    try {
        const response = await fetch('/register', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Registration successful! Please login.', 'success');
            showLogin();
            form.reset();
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Registration failed. Please try again.', 'error');
        console.error('Registration error:', error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Register';
    }
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Dashboard Functions
function initDashboard() {
    loadAttendanceStats();
    
    // Refresh attendance data every 30 seconds
    setInterval(loadAttendanceStats, 30000);
    
    // Add click handlers for interactive elements
    addDashboardInteractivity();
}

function addDashboardInteractivity() {
    // Add hover effects to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

async function loadAttendanceStats() {
    try {
        const response = await fetch('/get_attendance_stats');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading stats:', data.error);
            return;
        }
        
        updateAttendanceChart(data.daily_attendance);
        updateStatsDisplay(data);
        
    } catch (error) {
        console.error('Error loading attendance stats:', error);
        showNotification('Failed to load attendance statistics', 'error');
    }
}

function updateStatsDisplay(data) {
    const statsContainer = document.getElementById('attendanceStats');
    if (statsContainer && data.total_records !== undefined) {
        const totalElement = statsContainer.querySelector('p');
        if (totalElement) {
            totalElement.textContent = `Total Records: ${data.total_records}`;
        }
    }
}

function updateAttendanceChart(dailyData) {
    const canvas = document.getElementById('attendanceChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Prepare data (last 7 days)
    const dates = [];
    const counts = [];
    const today = new Date();
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        const dateStr = date.toISOString().split('T')[0];
        
        dates.push(date.toLocaleDateString('en-US', { weekday: 'short' }));
        counts.push(dailyData[dateStr] || 0);
    }
    
    // Draw chart
    drawBarChart(ctx, canvas, dates, counts);
}

function drawBarChart(ctx, canvas, labels, data) {
    const padding = 40;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    const maxValue = Math.max(...data, 1);
    const barWidth = chartWidth / labels.length;
    
    // Set styles
    ctx.fillStyle = '#667eea';
    ctx.strokeStyle = '#333';
    ctx.font = '12px Arial';
    
    // Draw bars
    for (let i = 0; i < data.length; i++) {
        const barHeight = (data[i] / maxValue) * chartHeight;
        const x = padding + i * barWidth + barWidth * 0.1;
        const y = canvas.height - padding - barHeight;
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth * 0.8, barHeight);
        
        // Draw labels
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + barWidth * 0.4, canvas.height - padding + 15);
        
        // Draw values
        if (data[i] > 0) {
            ctx.fillText(data[i], x + barWidth * 0.4, y - 5);
        }
        
        ctx.fillStyle = '#667eea';
    }
}

// Face Registration Functions
function initFaceRegistration() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('faceImage');
    const form = document.getElementById('faceRegisterForm');
    
    // Tab switching functionality
    const uploadTab = document.getElementById('uploadTab');
    const cameraTab = document.getElementById('cameraTab');
    const uploadContent = document.getElementById('uploadContent');
    const cameraContent = document.getElementById('cameraContent');
    
    if (uploadTab && cameraTab) {
        uploadTab.addEventListener('click', () => {
            uploadTab.classList.add('active');
            cameraTab.classList.remove('active');
            uploadContent.classList.add('active');
            cameraContent.classList.remove('active');
            stopVideoStream(); // Stop camera when switching to upload tab
        });
        
        cameraTab.addEventListener('click', () => {
            cameraTab.classList.add('active');
            uploadTab.classList.remove('active');
            cameraContent.classList.add('active');
            uploadContent.classList.remove('active');
            startVideoStream(); // Start camera when switching to camera tab
        });
    }
    
    if (uploadArea && fileInput) {
        // Handle drag and drop
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Handle file selection
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (form) {
        form.addEventListener('submit', handleFaceRegistration);
    }
    
    // Camera capture functionality
    const captureBtn = document.getElementById('captureBtn');
    const registerCapturedBtn = document.getElementById('registerCapturedBtn');
    
    if (captureBtn) {
        captureBtn.addEventListener('click', capturePhoto);
    }
    
    if (registerCapturedBtn) {
        registerCapturedBtn.addEventListener('click', registerCapturedPhoto);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    const uploadArea = event.currentTarget;
    uploadArea.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('faceImage').files = files;
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showNotification('Please select an image file.', 'error');
            return;
        }
        
        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showNotification('File size must be less than 5MB.', 'error');
            return;
        }
        
        // Show preview
        showImagePreview(file);
    }
}

function showImagePreview(file) {
    const previewContainer = document.getElementById('imagePreview');
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewContainer.innerHTML = `
            <div class="image-preview fade-in">
                <img src="${e.target.result}" alt="Preview">
                <p>Selected: ${file.name}</p>
                <p>Size: ${formatFileSize(file.size)}</p>
            </div>
        `;
    };
    
    reader.onerror = function() {
        showNotification('Error reading file', 'error');
    };
    
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Video stream variables
let videoStream = null;
let capturedImageData = null;

// Start video stream from camera
function startVideoStream() {
    const video = document.getElementById('videoElement');
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Stop any existing stream
        stopVideoStream();
        
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                videoStream = stream;
                video.srcObject = stream;
                video.play();
            })
            .catch(function(error) {
                console.error('Error accessing camera:', error);
                showNotification('Could not access camera. Please check permissions.', 'error');
            });
    } else {
        showNotification('Your browser does not support camera access.', 'error');
    }
}

// Stop video stream
function stopVideoStream() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => {
            track.stop();
        });
        videoStream = null;
        
        const video = document.getElementById('videoElement');
        if (video) {
            video.srcObject = null;
        }
    }
}

// Capture photo from video stream
function capturePhoto() {
    const video = document.getElementById('videoElement');
    const capturedImagePreview = document.getElementById('capturedImagePreview');
    const registerCapturedBtn = document.getElementById('registerCapturedBtn');
    const captureBtn = document.getElementById('captureBtn');
    
    if (!videoStream) {
        showNotification('Camera is not active. Please allow camera access.', 'error');
        return;
    }
    
    // Create a canvas element to capture the current video frame
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to data URL (base64 encoded image)
    capturedImageData = canvas.toDataURL('image/jpeg');
    
    // Display the captured image
    capturedImagePreview.innerHTML = `
        <div class="image-preview fade-in">
            <img src="${capturedImageData}" alt="Captured Photo">
            <p>Photo captured successfully</p>
        </div>
    `;
    
    // Show the register button
    registerCapturedBtn.style.display = 'block';
    captureBtn.textContent = 'Capture Again';
}

// Register the captured photo
async function registerCapturedPhoto() {
    const registerCapturedBtn = document.getElementById('registerCapturedBtn');
    const messageDiv = document.getElementById('message');
    
    if (!capturedImageData) {
        showNotification('Please capture a photo first.', 'error');
        return;
    }
    
    // Disable button and show loading
    registerCapturedBtn.disabled = true;
    registerCapturedBtn.innerHTML = '<span class="spinner"></span> Processing...';
    
    try {
        // Convert base64 to blob
        const response = await fetch(capturedImageData);
        const blob = await response.blob();
        
        // Create a File object from the blob
        const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('face_image', file);
        
        // Send to server
        const registerResponse = await fetch('/register_face', {
            method: 'POST',
            body: formData
        });
        
        const result = await registerResponse.json();
        
        if (result.success) {
            showNotification(result.message, 'success');
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 2000);
        } else {
            showNotification(result.message, 'error');
            registerCapturedBtn.disabled = false;
            registerCapturedBtn.innerHTML = 'Register Face';
        }
    } catch (error) {
        console.error('Error registering face:', error);
        showNotification('Failed to register face. Please try again.', 'error');
        registerCapturedBtn.disabled = false;
        registerCapturedBtn.innerHTML = 'Register Face';
    }
}

async function handleFaceRegistration(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('faceImage');
    const registerBtn = document.getElementById('registerBtn');
    const messageDiv = document.getElementById('message');
    
    if (!fileInput.files[0]) {
        showNotification('Please select an image first.', 'error');
        return;
    }
    
    // Disable button and show loading
    registerBtn.disabled = true;
    registerBtn.innerHTML = '<span class="spinner"></span> Processing...';
    
    const formData = new FormData();
    formData.append('face_image', fileInput.files[0]);
    
    try {
        const response = await fetch('/register_face', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification(result.message, 'success');
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 2000);
        } else {
            showNotification(result.message, 'error');
        }
    } catch (error) {
        showNotification('Registration failed. Please try again.', 'error');
        console.error('Face registration error:', error);
    } finally {
        registerBtn.disabled = false;
        registerBtn.innerHTML = 'Register Face';
    }
}

// Attendance Functions
function initAttendancePage() {
    const markBtn = document.getElementById('markAttendanceBtn');
    if (markBtn) {
        markBtn.addEventListener('click', markAttendance);
    }
    
    // Start recognition status updates
    startRecognitionStatus();
    
    // Monitor video feed
    monitorVideoFeed();
    
    // Initialize tabs
    initAttendanceTabs();
    
    // Initialize camera capture functionality
    initCameraCapture();
}

function startRecognitionStatus() {
    const statusElement = document.getElementById('recognitionStatus');
    if (!statusElement) return;
    
    let statusMessages = [
        'Scanning for faces...',
        'Processing camera feed...',
        'Ready for attendance...',
        'Waiting for recognition...'
    ];
    
    let messageIndex = 0;
    
    recognitionInterval = setInterval(() => {
        statusElement.textContent = statusMessages[messageIndex];
        statusElement.classList.add('pulse');
        
        setTimeout(() => {
            statusElement.classList.remove('pulse');
        }, 1000);
        
        messageIndex = (messageIndex + 1) % statusMessages.length;
    }, 3000);
}

function monitorVideoFeed() {
    const videoElement = document.getElementById('videoFeed');
    if (!videoElement) return;
    
    videoElement.addEventListener('load', function() {
        videoFeedActive = true;
        console.log('Video feed active');
    });
    
    videoElement.addEventListener('error', function() {
        videoFeedActive = false;
        showNotification('Camera feed error. Please check camera permissions.', 'error');
    });
}

function initAttendanceTabs() {
    const liveTab = document.getElementById('liveTab');
    const captureTab = document.getElementById('captureTab');
    const liveContent = document.getElementById('liveContent');
    const captureContent = document.getElementById('captureContent');
    
    if (!liveTab || !captureTab) return;
    
    liveTab.addEventListener('click', () => {
        liveTab.classList.add('active');
        captureTab.classList.remove('active');
        liveContent.classList.add('active');
        captureContent.classList.remove('active');
        
        // Stop camera stream when switching to live feed tab
        stopCameraStream();
    });
    
    captureTab.addEventListener('click', () => {
        captureTab.classList.add('active');
        liveTab.classList.remove('active');
        captureContent.classList.add('active');
        liveContent.classList.remove('active');
        
        // Start camera stream when switching to capture tab
        startCameraStream();
    });
}

// Camera capture variables
let cameraStream = null;

function initCameraCapture() {
    const captureBtn = document.getElementById('captureBtn');
    const markAttendanceCapturedBtn = document.getElementById('markAttendanceCapturedBtn');
    
    if (captureBtn) {
        captureBtn.addEventListener('click', capturePhoto);
    }
    
    if (markAttendanceCapturedBtn) {
        markAttendanceCapturedBtn.addEventListener('click', markAttendanceWithCapturedPhoto);
    }
}

async function markAttendance() {
    const markBtn = document.getElementById('markAttendanceBtn');
    const messageDiv = document.getElementById('attendanceMessage');
    
    if (!videoFeedActive) {
        showNotification('Camera feed not active. Please ensure camera is working.', 'error');
        return;
    }
    
    // Disable button
    markBtn.disabled = true;
    markBtn.innerHTML = '<span class="spinner"></span> Processing...';
    
    try {
        const response = await fetch('/mark_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showAttendanceMessage(result.message, 'success');
            showNotification('Attendance marked successfully!', 'success');
            
            // Redirect to dashboard after successful attendance
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 3000);
        } else {
            showAttendanceMessage(result.message, 'error');
            showNotification(result.message, 'error');
        }
    } catch (error) {
        const errorMsg = 'Failed to mark attendance. Please try again.';
        showAttendanceMessage(errorMsg, 'error');
        showNotification(errorMsg, 'error');
        console.error('Attendance marking error:', error);
    } finally {
        // Re-enable button after 3 seconds
        setTimeout(() => {
            markBtn.disabled = false;
            markBtn.innerHTML = 'Mark My Attendance';
        }, 3000);
    }
}

function startCameraStream() {
    const videoElement = document.getElementById('videoElement');
    if (!videoElement) return;
    
    // Stop any existing stream
    stopCameraStream();
    
    // Access the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoElement.srcObject = stream;
            cameraStream = stream;
        })
        .catch(error => {
            console.error('Error accessing camera:', error);
            showNotification('Could not access camera. Please check permissions.', 'error');
        });
}

function stopCameraStream() {
    if (cameraStream) {
        const tracks = cameraStream.getTracks();
        tracks.forEach(track => track.stop());
        cameraStream = null;
        
        const videoElement = document.getElementById('videoElement');
        if (videoElement) {
            videoElement.srcObject = null;
        }
    }
}

function capturePhoto() {
    const videoElement = document.getElementById('videoElement');
    const canvas = document.getElementById('canvas');
    const capturedImage = document.getElementById('capturedImage');
    const captureBtn = document.getElementById('captureBtn');
    const markAttendanceCapturedBtn = document.getElementById('markAttendanceCapturedBtn');
    
    if (!videoElement || !canvas || !capturedImage) return;
    
    // Set canvas dimensions to match video
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    // Draw video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to image
    capturedImageData = canvas.toDataURL('image/jpeg');
    capturedImage.src = capturedImageData;
    capturedImage.style.display = 'block';
    
    // Update button states
    captureBtn.textContent = 'Retake Photo';
    markAttendanceCapturedBtn.disabled = false;
}

async function markAttendanceWithCapturedPhoto() {
    if (!capturedImageData) {
        showNotification('Please capture a photo first.', 'error');
        return;
    }
    
    const markAttendanceCapturedBtn = document.getElementById('markAttendanceCapturedBtn');
    
    // Disable button and show loading
    markAttendanceCapturedBtn.disabled = true;
    markAttendanceCapturedBtn.innerHTML = '<span class="spinner"></span> Processing...';
    
    try {
        // Convert base64 to blob
        const response = await fetch(capturedImageData);
        const blob = await response.blob();
        
        // Create a File object from the blob
        const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
        
        const formData = new FormData();
        formData.append('face_image', file);
        
        // Send to server
        const attendanceResponse = await fetch('/mark_attendance_with_photo', {
            method: 'POST',
            body: formData
        });
        
        const result = await attendanceResponse.json();
        
        if (result.success) {
            showNotification(result.message, 'success');
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 2000);
        } else {
            showNotification(result.message, 'error');
            markAttendanceCapturedBtn.disabled = false;
            markAttendanceCapturedBtn.innerHTML = 'Mark Attendance';
        }
    } catch (error) {
        console.error('Error marking attendance:', error);
        showNotification('Failed to mark attendance. Please try again.', 'error');
        markAttendanceCapturedBtn.disabled = false;
        markAttendanceCapturedBtn.innerHTML = 'Mark Attendance';
    }
}

// Utility Functions
function showMessage(message, type) {
    const messageDiv = document.getElementById('message') || createMessageDiv();
    messageDiv.innerHTML = `<div class="${type} fade-in">${message}</div>`;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        messageDiv.innerHTML = '';
    }, 5000);
}

function showAttendanceMessage(message, type) {
    const messageDiv = document.getElementById('attendanceMessage');
    if (messageDiv) {
        messageDiv.innerHTML = `<div class="${type} fade-in">${message}</div>`;
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            messageDiv.innerHTML = '';
        }, 5000);
    }
}

// Clean up resources when leaving the page
window.addEventListener('beforeunload', () => {
    stopCameraStream();
});

function showNotification(message, type) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notif => notif.remove());
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
    
    // Allow manual dismiss by clicking
    notification.addEventListener('click', () => {
        notification.remove();
    });
}

function createMessageDiv() {
    const div = document.createElement('div');
    div.id = 'message';
    div.style.marginTop = '1rem';
    
    // Find appropriate container
    const container = document.querySelector('.card') || document.querySelector('.container') || document.body;
    container.appendChild(div);
    return div;
}

// Form validation helpers
function validateForm(formData, requiredFields) {
    const errors = [];
    
    requiredFields.forEach(field => {
        const value = formData.get(field);
        if (!value || value.trim() === '') {
            errors.push(`${field.replace('_', ' ')} is required`);
        }
    });
    
    return errors;
}

function sanitizeInput(input) {
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
}

// Camera and video utilities
function checkCameraPermissions() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                // Stop the stream immediately as we just wanted to check permissions
                stream.getTracks().forEach(track => track.stop());
                return true;
            })
            .catch(error => {
                console.error('Camera permission denied:', error);
                return false;
            });
    }
    return Promise.resolve(false);
}

// Local storage utilities (for storing user preferences)
function saveUserPreference(key, value) {
    try {
        localStorage.setItem(`attendance_${key}`, JSON.stringify(value));
    } catch (error) {
        console.warn('Unable to save preference:', error);
    }
}

function getUserPreference(key, defaultValue = null) {
    try {
        const stored = localStorage.getItem(`attendance_${key}`);
        return stored ? JSON.parse(stored) : defaultValue;
    } catch (error) {
        console.warn('Unable to get preference:', error);
        return defaultValue;
    }
}

// Performance monitoring
function measurePerformance(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    console.log(`${name} took ${(end - start).toFixed(2)} milliseconds`);
    return result;
}

// Debounce utility for preventing rapid API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle utility for limiting function execution
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Enhanced error handling
function handleApiError(error, context = '') {
    console.error(`API Error ${context}:`, error);
    
    let message = 'An unexpected error occurred';
    
    if (error.name === 'NetworkError' || error.message.includes('fetch')) {
        message = 'Network connection error. Please check your internet connection.';
    } else if (error.status === 401) {
        message = 'Session expired. Please login again.';
        setTimeout(() => {
            window.location.href = '/login';
        }, 2000);
    } else if (error.status === 403) {
        message = 'Access denied. You do not have permission to perform this action.';
    } else if (error.status === 500) {
        message = 'Server error. Please try again later.';
    }
    
    showNotification(message, 'error');
}

// Advanced chart drawing with animations
function drawAnimatedBarChart(ctx, canvas, labels, data, animationProgress = 1) {
    const padding = 40;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;
    const maxValue = Math.max(...data, 1);
    const barWidth = chartWidth / labels.length;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set styles
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
        
        // Draw y-axis labels
        ctx.fillStyle = '#666';
        ctx.textAlign = 'right';
        const value = Math.round(maxValue * (5 - i) / 5);
        ctx.fillText(value.toString(), padding - 10, y + 4);
    }
    
    // Draw bars with animation
    for (let i = 0; i < data.length; i++) {
        const barHeight = (data[i] / maxValue) * chartHeight * animationProgress;
        const x = padding + i * barWidth + barWidth * 0.1;
        const y = canvas.height - padding - barHeight;
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');
        
        // Draw bar
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth * 0.8, barHeight);
        
        // Draw bar border
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, barWidth * 0.8, barHeight);
        
        // Draw labels
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.fillText(labels[i], x + barWidth * 0.4, canvas.height - padding + 15);
        
        // Draw values on bars
        if (data[i] > 0 && animationProgress > 0.8) {
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(data[i], x + barWidth * 0.4, y + 20);
        }
    }
    
    // Draw title
    ctx.fillStyle = '#333';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Attendance Over Last 7 Days', canvas.width / 2, 25);
}

// Animation helper for charts
function animateChart(canvas, labels, data) {
    const ctx = canvas.getContext('2d');
    let progress = 0;
    const duration = 1000; // 1 second
    const startTime = Date.now();
    
    function animate() {
        const elapsed = Date.now() - startTime;
        progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out)
        const easedProgress = 1 - Math.pow(1 - progress, 3);
        
        drawAnimatedBarChart(ctx, canvas, labels, data, easedProgress);
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    animate();
}

// Enhanced attendance stats loading with animation
async function loadAttendanceStatsWithAnimation() {
    try {
        const response = await fetch('/get_attendance_stats');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading stats:', data.error);
            return;
        }
        
        // Update chart with animation
        updateAttendanceChartAnimated(data.daily_attendance);
        updateStatsDisplay(data);
        
    } catch (error) {
        handleApiError(error, 'loading attendance stats');
    }
}

function updateAttendanceChartAnimated(dailyData) {
    const canvas = document.getElementById('attendanceChart');
    if (!canvas) return;
    
    // Prepare data (last 7 days)
    const dates = [];
    const counts = [];
    const today = new Date();
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        const dateStr = date.toISOString().split('T')[0];
        
        dates.push(date.toLocaleDateString('en-US', { weekday: 'short' }));
        counts.push(dailyData[dateStr] || 0);
    }
    
    // Animate chart
    animateChart(canvas, dates, counts);
}

// Real-time updates for attendance page
function startRealTimeUpdates() {
    if (getCurrentPage() === 'attendance') {
        setInterval(updateAttendanceStatus, 5000); // Update every 5 seconds
    }
}

async function updateAttendanceStatus() {
    try {
        const response = await fetch('/get_current_status');
        if (response.ok) {
            const data = await response.json();
            
            const statusElement = document.getElementById('recognitionStatus');
            if (statusElement && data.status) {
                statusElement.textContent = data.status;
            }
        }
    } catch (error) {
        // Silently handle errors for real-time updates
        console.warn('Failed to update attendance status:', error);
    }
}

// Cleanup function
function cleanup() {
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }
    
    // Clear any other intervals or timeouts
    const highestTimeoutId = setTimeout(";");
    for (let i = 0; i < highestTimeoutId; i++) {
        clearTimeout(i);
    }
}

// Event listeners for page lifecycle
window.addEventListener('beforeunload', cleanup);

window.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, pause updates
        cleanup();
    } else {
        // Page is visible, resume updates
        initializePage();
    }
});

// Handle network errors gracefully
window.addEventListener('offline', function() {
    showNotification('You are offline. Some features may not work.', 'error');
});

window.addEventListener('online', function() {
    showNotification('Connection restored.', 'success');
    // Retry any failed operations
    initializePage();
});

// Performance optimization
function optimizeImages() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        if (img.loading !== 'lazy') {
            img.loading = 'lazy';
        }
    });
}

// Initialize performance optimizations
document.addEventListener('DOMContentLoaded', function() {
    optimizeImages();
    startRealTimeUpdates();
});

// Export functions for testing (if in development mode)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateForm,
        sanitizeInput,
        formatFileSize,
        debounce,
        throttle
    };
}