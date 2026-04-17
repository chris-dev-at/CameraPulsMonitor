const ROI_LANDMARKS = [10, 67, 103, 104, 108, 151, 337, 332, 297, 338];
let bufferSize = 150;
const MAX_HISTORY = 10;
const BUFFER_SIZES = [90, 150, 240, 360];
const WARNING_MIN_DURATION = 500;
const FACE_SIZE_MIN = 0.15;

let faceMesh;
let camera;
let dataBuffer = [];
let times = [];
let bpm = 0;
let bpmHistory = [];
let bpmStdDev = 0;
let isProcessing = false;
let lastFrameTime = 0;
let fftData = [];
let fftFreqs = [];
let peakFreq = 0;
let signalSNR = 0;
let confidence = 0;
let faceAlignment = 1;
let faceSize = 0;
let showEnhancement = false;
let showMesh = true;
let showCamera = true;
let enhancedImageData = null;
let lastLandmarks = null;
let warningStartTime = 0;

const videoElement = document.getElementById('videoElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const graphCanvas = document.getElementById('graphCanvas');
const fftCanvas = document.getElementById('fftCanvas');
const enhancedCanvas = document.getElementById('enhancedCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const graphCtx = graphCanvas.getContext('2d');
const fftCtx = fftCanvas.getContext('2d');
const enhancedCtx = enhancedCanvas.getContext('2d');
const enhanceBtn = document.getElementById('enhanceBtn');
const meshBtn = document.getElementById('meshBtn');
const flashBtn = document.getElementById('flashBtn');

const bpmValueEl = document.getElementById('bpmValue');
const statusBadgeEl = document.getElementById('statusBadge');
const statusTextEl = document.getElementById('statusText');
const bufferProgressEl = document.getElementById('bufferProgress');
const bufferDurationEl = document.getElementById('bufferDuration');
const qualityValueEl = document.getElementById('qualityValue');
const startBtn = document.getElementById('startBtn');
const mainApp = document.getElementById('mainApp');
const bufferBtn = document.getElementById('bufferBtn');
const themeBtn = document.getElementById('themeBtn');

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    const themeIcon = document.getElementById('themeIcon');
    if (theme === 'light') {
        themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    } else {
        themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    }
}

function toggleTheme() {
    const current = localStorage.getItem('theme') || 'dark';
    setTheme(current === 'dark' ? 'light' : 'dark');
}

themeBtn.addEventListener('click', toggleTheme);

const infoBtn = document.getElementById('infoBtn');
const infoModal = document.getElementById('infoModal');
const closeInfoBtn = document.getElementById('closeInfoBtn');

if (infoBtn && infoModal) {
    infoBtn.addEventListener('click', () => {
        infoModal.classList.remove('hidden');
    });
    
    closeInfoBtn.addEventListener('click', () => {
        infoModal.classList.add('hidden');
    });
    
    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) {
            infoModal.classList.add('hidden');
        }
    });
}

function getGraphColors() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
        bg: isDark ? '#222222' : '#e8e8e8',
        line: isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
        text: isDark ? '#a0a0a0' : '#666666',
        primary: '#e85a6b'
    };
}

initTheme();

function numpyDetrend(data) {
    const x = Array.from({length: data.length}, (_, i) => i);
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += data[i];
        sumXY += x[i] * data[i];
        sumX2 += x[i] * x[i];
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return data.map((val, i) => val - (slope * x[i] + intercept));
}

function numpyBandpassFilter(data, fps, low = 0.75, high = 3.0) {
    const N = data.length;
    const real = new Array(N).fill(0);
    const imag = new Array(N).fill(0);
    
    for (let k = 0; k < N; k++) {
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real[k] += data[n] * Math.cos(angle);
            imag[k] -= data[n] * Math.sin(angle);
        }
    }
    
    const freqs = [];
    for (let k = 0; k <= Math.floor(N / 2); k++) {
        freqs.push(k * fps / N);
    }
    
    for (let k = 0; k < N; k++) {
        const freq = k * fps / N;
        if (freq < low || freq > high) {
            real[k] = 0;
            imag[k] = 0;
        }
    }
    
    const filtered = new Array(N).fill(0);
    for (let n = 0; n < N; n++) {
        let sumReal = 0, sumImag = 0;
        for (let k = 0; k < N; k++) {
            const angle = (2 * Math.PI * k * n) / N;
            sumReal += real[k] * Math.cos(angle) - imag[k] * Math.sin(angle);
            sumImag += real[k] * Math.sin(angle) + imag[k] * Math.cos(angle);
        }
        filtered[n] = sumReal / N;
    }
    
    return filtered;
}

function calculateBPM() {
    if (dataBuffer.length < bufferSize) return;
    
    const data = [...dataBuffer];
    const timesArr = [...times];
    
    const duration = timesArr[timesArr.length - 1] - timesArr[0];
    if (duration <= 0) return;
    
    const fps = timesArr.length / duration;
    
    let detrended = numpyDetrend(data);
    const mean = detrended.reduce((a, b) => a + b, 0) / detrended.length;
    const std = Math.sqrt(detrended.reduce((a, b) => a + (b - mean) ** 2, 0) / detrended.length);
    detrended = detrended.map(v => (v - mean) / (std + 1e-6));
    
    const filtered = numpyBandpassFilter(detrended, fps);
    
    const N = Math.max(filtered.length, 1024);
    const fftVals = [];
    for (let k = 0; k <= Math.floor(N / 2); k++) {
        let real = 0, imag = 0;
        for (let n = 0; n < filtered.length; n++) {
            const angle = (2 * Math.PI * k * n) / N;
            real += filtered[n] * Math.cos(angle);
            imag -= filtered[n] * Math.sin(angle);
        }
        fftVals.push(Math.sqrt(real * real + imag * imag));
    }
    
    const freqs = [];
    for (let k = 0; k <= Math.floor(N / 2); k++) {
        freqs.push(k * fps / N);
    }
    
    const validIndices = [];
    for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] >= 0.75 && freqs[i] <= 3.0) {
            validIndices.push(i);
        }
    }
    
    if (validIndices.length > 0) {
        let maxIdx = validIndices[0];
        let maxVal = fftVals[validIndices[0]];
        for (const idx of validIndices) {
            if (fftVals[idx] > maxVal) {
                maxVal = fftVals[idx];
                maxIdx = idx;
            }
        }
        
        peakFreq = freqs[maxIdx];
        const newBpm = peakFreq * 60;
        bpmHistory.push(newBpm);
        
        if (bpmHistory.length > MAX_HISTORY) {
            bpmHistory.shift();
        }
        
        bpm = bpmHistory.reduce((a, b) => a + b, 0) / bpmHistory.length;
        
        const mean = bpmHistory.reduce((a, b) => a + b, 0) / bpmHistory.length;
        const variance = bpmHistory.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / bpmHistory.length;
        bpmStdDev = Math.sqrt(variance);
        
        let noiseSum = 0;
        let noiseCount = 0;
        const windowSize = 10;
        
        for (let i = 0; i < validIndices.length; i++) {
            if (Math.abs(validIndices[i] - maxIdx) > windowSize) {
                noiseSum += fftVals[validIndices[i]];
                noiseCount++;
            }
        }
        
        const avgNoise = noiseCount > 0 ? noiseSum / noiseCount : 1;
        signalSNR = avgNoise > 0 ? maxVal / avgNoise : 1;
        signalSNR = Math.min(10, Math.max(0, signalSNR));
        
        const snrConf = Math.min(1, signalSNR / 2);
        const stabilityConf = bpmStdDev < 3 ? 1 : (bpmStdDev < 8 ? 0.5 : 0);
        const rangeConf = (bpm >= 55 && bpm <= 100) ? 1 : ((bpm >= 45 && bpm <= 170) ? 0.5 : 0);
        confidence = (snrConf * 0.5 + stabilityConf * 0.3 + rangeConf * 0.2) * 100;
        
        fftData = [...fftVals];
        fftFreqs = [...freqs];
    }
}

function getROIAverage(landmarks) {
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    
    if (!w || !h) return 0;
    
    const roiPoints = ROI_LANDMARKS.map(idx => ({
        x: Math.round(landmarks[idx].x * w),
        y: Math.round(landmarks[idx].y * h)
    }));
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(videoElement, 0, 0);
    const frameData = tempCtx.getImageData(0, 0, w, h).data;
    
    tempCtx.clearRect(0, 0, w, h);
    tempCtx.beginPath();
    tempCtx.moveTo(roiPoints[0].x, roiPoints[0].y);
    for (let i = 1; i < roiPoints.length; i++) {
        tempCtx.lineTo(roiPoints[i].x, roiPoints[i].y);
    }
    tempCtx.closePath();
    tempCtx.fill();
    
    const maskData = tempCtx.getImageData(0, 0, w, h).data;
    
    let sumGreen = 0;
    let count = 0;
    
    for (let i = 3; i < maskData.length; i += 4) {
        if (maskData[i] > 0) {
            const pixelIdx = (i / 4) * 4;
            sumGreen += frameData[pixelIdx + 1];
            count++;
        }
    }
    
    return count > 0 ? sumGreen / count : 0;
}

function calculateFaceAlignment(landmarks) {
    const nose = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const forehead = landmarks[10];
    const chin = landmarks[152];
    const leftEar = landmarks[234];
    const rightEar = landmarks[454];
    
    const eyeCenter = {
        x: (leftEye.x + rightEye.x) / 2,
        y: (leftEye.y + rightEye.y) / 2
    };
    
    const noseToEyeDiff = nose.x - eyeCenter.x;
    const faceWidth = rightEar.x - leftEar.x;
    const yaw = Math.abs(noseToEyeDiff) / (faceWidth / 2);
    
    const foreheadToChin = chin.y - forehead.y;
    const noseToForehead = nose.y - forehead.y;
    const pitch = Math.abs(noseToForehead - foreheadToChin / 2) / (foreheadToChin / 2);
    
    const leftEyeY = leftEye.y;
    const rightEyeY = rightEye.y;
    const roll = Math.abs(leftEyeY - rightEyeY) / (Math.abs(leftEye.x - rightEye.x));
    
    const alignment = Math.max(0, 1 - (yaw * 1.5 + pitch * 0.5 + roll * 0.3));
    return Math.min(1, Math.max(0, alignment));
}

function calculateFaceSize(landmarks) {
    const chin = landmarks[152];
    const forehead = landmarks[10];
    const leftEar = landmarks[234];
    const rightEar = landmarks[454];
    
    const faceHeight = Math.abs(chin.y - forehead.y);
    const faceWidth = Math.abs(rightEar.x - leftEar.x);
    
    return (faceHeight + faceWidth) / 2;
}

function drawOverlay(landmarks) {
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    
    if (!w || !h) return;
    
    overlayCanvas.width = w;
    overlayCanvas.height = h;
    overlayCtx.clearRect(0, 0, w, h);
    
    if (!showMesh) {
        lastLandmarks = landmarks;
        return;
    }
    
    overlayCtx.strokeStyle = '#00ff88';
    overlayCtx.lineWidth = 3;
    overlayCtx.fillStyle = 'rgba(0, 255, 136, 0.15)';
    
    overlayCtx.beginPath();
    const firstPt = landmarks[ROI_LANDMARKS[0]];
    overlayCtx.moveTo(firstPt.x * w, firstPt.y * h);
    
    for (let i = 1; i < ROI_LANDMARKS.length; i++) {
        const pt = landmarks[ROI_LANDMARKS[i]];
        overlayCtx.lineTo(pt.x * w, pt.y * h);
    }
    overlayCtx.closePath();
    overlayCtx.fill();
    overlayCtx.stroke();
    
    for (const idx of ROI_LANDMARKS) {
        const pt = landmarks[idx];
        overlayCtx.beginPath();
        overlayCtx.arc(pt.x * w, pt.y * h, 4, 0, 2 * Math.PI);
        overlayCtx.fillStyle = 'var(--primary-red)';
        overlayCtx.fill();
    }
    
    lastLandmarks = landmarks;
}

function drawEnhancedView() {
    try {
        if (!showEnhancement || !lastLandmarks || dataBuffer.length === 0) {
            enhancedCtx.clearRect(0, 0, enhancedCanvas.width || 640, enhancedCanvas.height || 480);
            return;
        }
        
        const w = videoElement.videoWidth;
        const h = videoElement.videoHeight;
        
        if (!w || !h) return;
        
        enhancedCanvas.width = w;
        enhancedCanvas.height = h;
        
        enhancedCtx.drawImage(videoElement, 0, 0);
        
        const roiPoints = ROI_LANDMARKS.map(idx => ({
            x: Math.round(lastLandmarks[idx].x * w),
            y: Math.round(lastLandmarks[idx].y * h)
        }));
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w;
        tempCanvas.height = h;
        const tempCtx = tempCanvas.getContext('2d');
        
        const normalizedSignal = (dataBuffer[dataBuffer.length - 1] - Math.min(...dataBuffer)) / 
                                (Math.max(...dataBuffer) - Math.min(...dataBuffer) || 1);
        
        const amplification = 3;
        const normalizedAmp = 1 + (normalizedSignal - 0.5) * amplification;
        
        tempCtx.drawImage(videoElement, 0, 0);
        
        const imageData = tempCtx.getImageData(0, 0, w, h);
        const data = imageData.data;
        
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = w;
        maskCanvas.height = h;
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.beginPath();
        maskCtx.moveTo(roiPoints[0].x, roiPoints[0].y);
        for (let i = 1; i < roiPoints.length; i++) {
            maskCtx.lineTo(roiPoints[i].x, roiPoints[i].y);
        }
        maskCtx.closePath();
        maskCtx.fill();
        const maskData = maskCtx.getImageData(0, 0, w, h).data;
        
        for (let i = 3; i < maskData.length; i += 4) {
            if (maskData[i] > 0) {
                const pixelIdx = i - 3;
                const r = data[pixelIdx];
                const g = data[pixelIdx + 1];
                const b = data[pixelIdx + 2];
                
                const greenDiff = g - 128;
                const enhancedGreen = Math.min(255, Math.max(0, 128 + greenDiff * normalizedAmp));
                const pulseEffect = Math.round((normalizedSignal - 0.5) * 80);
                
                data[pixelIdx] = Math.min(255, Math.max(0, r * 0.5 + pulseEffect));
                data[pixelIdx + 1] = Math.min(255, Math.max(0, enhancedGreen + pulseEffect));
                data[pixelIdx + 2] = Math.min(255, Math.max(0, b * 0.5));
            }
        }
        
        tempCtx.putImageData(imageData, 0, 0);
        
        enhancedCtx.drawImage(tempCanvas, 0, 0);
        
        enhancedCtx.strokeStyle = '#00d4ff';
        enhancedCtx.lineWidth = 3;
        enhancedCtx.beginPath();
        enhancedCtx.moveTo(roiPoints[0].x, roiPoints[0].y);
        for (let i = 1; i < roiPoints.length; i++) {
            enhancedCtx.lineTo(roiPoints[i].x, roiPoints[i].y);
        }
        enhancedCtx.closePath();
        enhancedCtx.stroke();
        
        enhancedCtx.save();
        enhancedCtx.scale(-1, 1);
        enhancedCtx.translate(-w, 0);
        
        enhancedCtx.fillStyle = 'rgba(200, 200, 200, 0.7)';
        enhancedCtx.fillRect(w - 190, 10, 180, 50);
        enhancedCtx.fillStyle = 'var(--primary-red)';
        enhancedCtx.font = 'bold 14px Inter';
        enhancedCtx.fillText('Signal: ' + Math.round(normalizedSignal * 100) + '%', w - 180, 32);
        enhancedCtx.fillStyle = 'var(--text-muted)';
        enhancedCtx.font = '12px Inter';
        enhancedCtx.fillText('Enhanced View', w - 180, 50);
        
        enhancedCtx.restore();
    } catch (e) {
        console.error('Enhanced view error:', e);
    }
}

function drawGraph() {
    const rect = graphCanvas.getBoundingClientRect();
    graphCanvas.width = rect.width * 2;
    graphCanvas.height = rect.height * 2;
    
    const w = graphCanvas.width;
    const h = graphCanvas.height;
    const colors = getGraphColors();
    
    graphCtx.fillStyle = colors.bg;
    graphCtx.fillRect(0, 0, w, h);
    
    graphCtx.strokeStyle = colors.line;
    graphCtx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {
        graphCtx.beginPath();
        graphCtx.moveTo(0, y);
        graphCtx.lineTo(w, y);
        graphCtx.stroke();
    }
    
    if (dataBuffer.length < 2) {
        graphCtx.fillStyle = colors.text;
        graphCtx.font = `${Math.round(w / 20)}px Inter`;
        graphCtx.textAlign = 'center';
        graphCtx.fillText('Collecting signal...', w / 2, h / 2);
        return;
    }
    
    const data = [...dataBuffer];
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const gradient = graphCtx.createLinearGradient(0, h, 0, 0);
    gradient.addColorStop(0, 'rgba(125, 211, 168, 0.3)');
    gradient.addColorStop(1, 'rgba(125, 211, 168, 0.7)');
    
    graphCtx.fillStyle = gradient;
    graphCtx.beginPath();
    graphCtx.moveTo(0, h);
    
    for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * w;
        const y = h - 20 - ((data[i] - min) / range) * (h - 40);
        graphCtx.lineTo(x, y);
    }
    
    graphCtx.lineTo(w, h);
    graphCtx.closePath();
    graphCtx.fill();
    
    const lineGradient = graphCtx.createLinearGradient(0, 0, w, 0);
    lineGradient.addColorStop(0, '#7dd3a8');
    lineGradient.addColorStop(1, '#5bc08a');
    graphCtx.strokeStyle = lineGradient;
    graphCtx.lineWidth = 3;
    graphCtx.lineCap = 'round';
    graphCtx.lineJoin = 'round';
    graphCtx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * w;
        const y = h - 20 - ((data[i] - min) / range) * (h - 40);
        
        if (i === 0) {
            graphCtx.moveTo(x, y);
        } else {
            graphCtx.lineTo(x, y);
        }
    }
    
    graphCtx.stroke();
    
    graphCtx.fillStyle = colors.primary;
    graphCtx.font = `bold ${Math.round(w / 30)}px Inter`;
    graphCtx.textAlign = 'left';
    graphCtx.fillText('Raw Signal', 15, 30);
}

function drawFFT() {
    const rect = fftCanvas.getBoundingClientRect();
    fftCanvas.width = rect.width * 2;
    fftCanvas.height = rect.height * 2;
    
    const w = fftCanvas.width;
    const h = fftCanvas.height;
    const colors = getGraphColors();
    
    fftCtx.fillStyle = colors.bg;
    fftCtx.fillRect(0, 0, w, h);
    
    fftCtx.strokeStyle = colors.line;
    fftCtx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {
        fftCtx.beginPath();
        fftCtx.moveTo(0, y);
        fftCtx.lineTo(w, y);
        fftCtx.stroke();
    }
    
    if (fftData.length === 0 || fftFreqs.length === 0) {
        fftCtx.fillStyle = colors.text;
        fftCtx.font = `${Math.round(w / 20)}px Inter`;
        fftCtx.textAlign = 'center';
        fftCtx.fillText('Processing FFT...', w / 2, h / 2);
        return;
    }
    
    const lowIdx = Math.max(0, fftFreqs.findIndex(f => f >= 0.5) - 1);
    const highIdx = fftFreqs.findIndex(f => f >= 4.0);
    const displaySlice = fftData.slice(lowIdx, highIdx > 0 ? highIdx + 1 : undefined);
    
    if (displaySlice.length === 0) return;
    
    const maxVal = Math.max(...displaySlice) || 1;
    
    const gradient = fftCtx.createLinearGradient(0, h, 0, 0);
    gradient.addColorStop(0, 'rgba(184, 160, 216, 0.3)');
    gradient.addColorStop(1, 'rgba(184, 160, 216, 0.7)');
    
    fftCtx.fillStyle = gradient;
    fftCtx.beginPath();
    fftCtx.moveTo(0, h);
    
    for (let i = 0; i < displaySlice.length; i++) {
        const x = (i / (displaySlice.length - 1)) * w;
        const y = h - 20 - (displaySlice[i] / maxVal) * (h - 40);
        fftCtx.lineTo(x, y);
    }
    
    fftCtx.lineTo(w, h);
    fftCtx.closePath();
    fftCtx.fill();
    
    const fftLineGradient = fftCtx.createLinearGradient(0, 0, w, 0);
    fftLineGradient.addColorStop(0, '#b8a0d8');
    fftLineGradient.addColorStop(1, '#9b80c4');
    fftCtx.strokeStyle = fftLineGradient;
    fftCtx.lineWidth = 3;
    fftCtx.beginPath();
    
    for (let i = 0; i < displaySlice.length; i++) {
        const x = (i / (displaySlice.length - 1)) * w;
        const y = h - 20 - (displaySlice[i] / maxVal) * (h - 40);
        
        if (i === 0) {
            fftCtx.moveTo(x, y);
        } else {
            fftCtx.lineTo(x, y);
        }
    }
    fftCtx.stroke();
    
    if (peakFreq > 0) {
        const peakIdx = fftFreqs.findIndex(f => f >= peakFreq);
        if (peakIdx >= 0 && peakIdx < fftData.length) {
            const relativeIdx = peakIdx - lowIdx;
            if (relativeIdx >= 0 && relativeIdx < displaySlice.length) {
                const px = (relativeIdx / (displaySlice.length - 1)) * w;
                const py = h - 20 - (fftData[peakIdx] / maxVal) * (h - 40);
                
                fftCtx.shadowColor = colors.primary;
                fftCtx.shadowBlur = 15;
                fftCtx.fillStyle = colors.primary;
                fftCtx.beginPath();
                fftCtx.arc(px, py, 8, 0, 2 * Math.PI);
                fftCtx.fill();
                fftCtx.shadowBlur = 0;
            }
        }
    }
    
    fftCtx.fillStyle = colors.primary;
    fftCtx.font = `bold ${Math.round(w / 30)}px Inter`;
    fftCtx.textAlign = 'left';
    fftCtx.fillText('FFT Spectrum', 15, 30);
}

function updateUI(faceFound = true) {
    const progress = (dataBuffer.length / bufferSize) * 100;
    const isConfident = confidence >= 50;
    const hasEnoughData = dataBuffer.length >= bufferSize;
    const now = Date.now();
    const isInWarningPeriod = warningStartTime > 0 && (now - warningStartTime) < WARNING_MIN_DURATION;
    
    if (!faceFound) {
        bpmValueEl.textContent = '--';
        
        if (measureMode === 'finger') {
            const brightness = getFingerBrightness();
            const fingerSignal = getFingerSignalValue();
            const fingerCovered = checkFingerCovered();
            
            if (fingerCovered && brightness >= 20 && brightness <= 250) {
                statusTextEl.textContent = 'Collecting data...';
            } else if (brightness < 20) {
                statusTextEl.textContent = 'More brightness needed';
            } else if (brightness > 250) {
                statusTextEl.textContent = 'Less brightness needed';
            } else {
                statusTextEl.textContent = 'Place finger on camera';
            }
        } else {
            statusTextEl.textContent = 'No face detected';
        }
        bpmValueEl.style.color = 'var(--primary-red)';
        statusTextEl.style.color = 'var(--primary-red)';
        statusBadgeEl.className = 'status-badge error';
        warningStartTime = 0;
        
        qualityValueEl.textContent = '--';
        bufferProgressEl.textContent = Math.round(progress) + '%';
        bufferDurationEl.textContent = Math.round(bufferSize / 30) + 's';
        return;
    }
    
    if (hasEnoughData) {
        bpmValueEl.textContent = Math.round(bpm);
    } else {
        bpmValueEl.textContent = '--';
    }
    
    const isAligned = faceAlignment >= 0.6;
    const isCloseEnough = faceSize >= FACE_SIZE_MIN;
    
    if (hasEnoughData) {
        
        if (isConfident && !isInWarningPeriod && isAligned) {
            statusTextEl.textContent = 'Measuring...';
            statusBadgeEl.className = 'status-badge';
            bpmValueEl.style.color = 'var(--text-dark)';
            statusTextEl.style.color = 'var(--text-dark)';
            warningStartTime = 0;
        } else {
            if (!isConfident && warningStartTime === 0) {
                warningStartTime = now;
            }
            if (!isCloseEnough && measureMode !== 'finger') {
                statusTextEl.textContent = 'Move closer to camera';
            } else if (!isAligned && measureMode !== 'finger') {
                statusTextEl.textContent = 'Look straight at camera';
            } else {
                statusTextEl.textContent = 'Bad signal';
            }
            statusBadgeEl.className = 'status-badge warning';
            bpmValueEl.style.color = 'var(--warning-orange)';
            statusTextEl.style.color = 'var(--warning-orange)';
        }
        
        qualityValueEl.textContent = Math.round(confidence) + '%';
    } else {
        if (!isCloseEnough && measureMode !== 'finger') {
            statusTextEl.textContent = 'Move closer to camera';
            statusBadgeEl.className = 'status-badge warning';
            bpmValueEl.style.color = 'var(--warning-orange)';
            statusTextEl.style.color = 'var(--warning-orange)';
        } else if (faceAlignment < 0.6 && measureMode !== 'finger') {
            statusTextEl.textContent = 'Look straight at camera';
            statusBadgeEl.className = 'status-badge warning';
            bpmValueEl.style.color = 'var(--warning-orange)';
            statusTextEl.style.color = 'var(--warning-orange)';
        } else {
            statusTextEl.textContent = `Collecting data... ${Math.round(progress)}%`;
            statusTextEl.style.color = 'var(--text-dark)';
            statusBadgeEl.className = 'status-badge warning';
            bpmValueEl.style.color = 'var(--text-muted)';
        }
        qualityValueEl.textContent = '--';
        warningStartTime = 0;
    }
    
    bufferProgressEl.textContent = Math.round(progress) + '%';
    bufferDurationEl.textContent = Math.round(bufferSize / 30) + 's';
    drawGraph();
    drawFFT();
    drawEnhancedView();
}

async function onResults(results) {
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    
    if (!w || !h) return;
    
    overlayCanvas.width = w;
    overlayCanvas.height = h;
    overlayCtx.clearRect(0, 0, w, h);
    
    let faceFound = false;
    
    if (measureMode === 'finger') {
        const brightness = getFingerBrightness();
        const fingerVal = getFingerSignalValue();
        
        if (fingerVal > 10 && brightness >= 20 && brightness <= 250) {
            dataBuffer.push(fingerVal);
            times.push(performance.now() / 1000);
            
            if (dataBuffer.length > bufferSize) {
                dataBuffer.shift();
                times.shift();
            }
            
            if (dataBuffer.length >= bufferSize) {
                calculateBPM();
            }
        }
        faceFound = dataBuffer.length > 0 && fingerVal > 10 && brightness >= 20 && brightness <= 250;
    } else if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        faceFound = true;
        const landmarks = results.multiFaceLandmarks[0];
        drawOverlay(landmarks);
        faceAlignment = calculateFaceAlignment(landmarks);
        faceSize = calculateFaceSize(landmarks);
        
        const greenVal = getROIAverage(landmarks);
        
        if (greenVal > 0) {
            dataBuffer.push(greenVal);
            times.push(performance.now() / 1000);
            
            if (dataBuffer.length > bufferSize) {
                dataBuffer.shift();
                times.shift();
            }
            
            if (dataBuffer.length >= bufferSize) {
                calculateBPM();
            }
        }
    }
    
    updateUI(faceFound);
}



function checkFingerCovered() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(videoElement, 0, 0, 100, 100);
    const frameData = tempCtx.getImageData(0, 0, 100, 100).data;
    
    let avgRed = 0;
    let avgGreen = 0;
    let avgBlue = 0;
    let count = 0;
    
    for (let i = 0; i < frameData.length; i += 4) {
        avgRed += frameData[i];
        avgGreen += frameData[i + 1];
        avgBlue += frameData[i + 2];
        count++;
    }
    
    avgRed /= count;
    avgGreen /= count;
    avgBlue /= count;
    
    let variance = 0;
    for (let i = 0; i < frameData.length; i += 4) {
        const r = frameData[i];
        const g = frameData[i + 1];
        const b = frameData[i + 2];
        const diff = Math.abs(r - avgRed) + Math.abs(g - avgGreen) + Math.abs(b - avgBlue);
        variance += diff * diff;
    }
    variance /= count;
    
    const brightness = (avgRed + avgGreen + avgBlue) / 3;
    const redness = avgRed - (avgGreen + avgBlue) / 2;
    
    const isUniform = variance < 2000;
    const isReddish = redness > 15;
    const isDim = brightness < 30;
    
    return (isUniform && isReddish) || (isDim && isReddish);
}

function getFingerBrightness() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(videoElement, 0, 0, 100, 100);
    const frameData = tempCtx.getImageData(0, 0, 100, 100).data;
    
    let avgRed = 0;
    let avgGreen = 0;
    let avgBlue = 0;
    let count = 0;
    
    for (let i = 0; i < frameData.length; i += 4) {
        avgRed += frameData[i];
        avgGreen += frameData[i + 1];
        avgBlue += frameData[i + 2];
        count++;
    }
    
    return (avgRed / count + avgGreen / count + avgBlue / count) / 3;
}

function getFingerSignalValue() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 100;
    tempCanvas.height = 100;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(videoElement, 0, 0, 100, 100);
    const frameData = tempCtx.getImageData(0, 0, 100, 100).data;
    
    let avgRed = 0;
    let avgGreen = 0;
    let avgBlue = 0;
    let count = 0;
    
    for (let i = 0; i < frameData.length; i += 4) {
        avgRed += frameData[i];
        avgGreen += frameData[i + 1];
        avgBlue += frameData[i + 2];
        count++;
    }
    
    avgRed /= count;
    avgGreen /= count;
    avgBlue /= count;
    
    return avgRed - (avgGreen + avgBlue) / 2;
}

async function initFaceMesh(mode = 'face') {
    statusTextEl.textContent = 'Checking camera...';
    statusBadgeEl.className = 'status-badge warning';
    
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not supported');
        }
        
        const facing = mode === 'finger' ? 'environment' : 'user';
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 }, 
                facingMode: facing
            } 
        });
        
        videoElement.srcObject = stream;
        await videoElement.play();
        
        await new Promise(resolve => {
            const check = () => {
                if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                    resolve();
                } else {
                    setTimeout(check, 50);
                }
            };
            check();
        });
        
        statusTextEl.textContent = 'Loading AI model...';
        
        faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        faceMesh.onResults(onResults);
        
        await faceMesh.initialize();
        
        statusTextEl.textContent = 'Camera ready - Face the camera';
        statusBadgeEl.className = 'status-badge';
        
        function processFrame(timestamp) {
            if (timestamp - lastFrameTime >= 33) {
                if (videoElement.readyState >= 2 && videoElement.videoWidth > 0 && !isProcessing) {
                    isProcessing = true;
                    try {
                        faceMesh.send({ image: videoElement });
                    } catch (e) {
                        console.warn('Frame error:', e);
                    }
                    isProcessing = false;
                }
                lastFrameTime = timestamp;
            }
            requestAnimationFrame(processFrame);
        }
        
        requestAnimationFrame(processFrame);
    } catch (err) {
        console.error('Init error:', err);
        throw err;
    }
}

const stopBtn = document.getElementById('stopBtn');
const measureSelect = document.querySelector('.measure-select');

let measureMode = 'face';

startBtn.addEventListener('click', async () => {
    console.log('Start button clicked');
    startBtn.disabled = true;
    startBtn.innerHTML = '<span>Loading AI Model...</span>';
    
    measureMode = document.querySelector('input[name="measure"]:checked').value;
    
    measureSelect.classList.add('hidden');
    bpmValueEl.style.color = 'var(--text-dark)';
    statusTextEl.style.color = 'var(--text-dark)';
    
    try {
        if (measureMode === 'mic') {
            startBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
            mainApp.classList.remove('hidden');
            
            const camSection = document.getElementById('cameraSection');
            if (camSection) camSection.classList.add('hidden');
            
            const graphsRow = document.querySelector('.graphs-row');
            if (graphsRow) graphsRow.classList.add('hidden');
            
            document.querySelectorAll('.row-center').forEach(el => el.classList.add('hidden'));
            
            const micViewEl = document.getElementById('micView');
            if (micViewEl) micViewEl.classList.remove('hidden');
        } else {
            await initFaceMesh(measureMode);
            startBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
            mainApp.classList.remove('hidden');
            
            if (measureMode === 'finger') {
                //document.getElementById('instructionsFace').classList.add('hidden');
                //document.getElementById('instructionsFinger').classList.remove('hidden');
                meshBtn.classList.add('hidden');
                cameraBtn.classList.add('hidden');
                enhanceBtn.classList.add('hidden');
                flashBtn.classList.remove('hidden');
            } else {
                document.getElementById('instructionsFace').classList.remove('hidden');
                document.getElementById('instructionsFinger').classList.add('hidden');
            }
        }
    } catch (err) {
        console.error('Init Error:', err);
        let errorMsg = 'Camera/AI failed to load';
        if (err.message && err.message.toLowerCase().includes('camera')) {
            errorMsg = 'Camera access denied';
        } else if (err.message && err.message.toLowerCase().includes('mediapipe')) {
            errorMsg = 'AI model failed to load';
        }
        startBtn.disabled = false;
        startBtn.innerHTML = '<span>TAP to Retry</span>';
        statusTextEl.textContent = errorMsg;
        statusBadgeEl.className = 'status-badge error';
    }
});

stopBtn.addEventListener('click', () => {
    location.reload();
});

enhanceBtn.addEventListener('click', () => {
    showEnhancement = !showEnhancement;
    enhanceBtn.classList.toggle('active', showEnhancement);
    if (!showEnhancement) {
        enhancedCtx.clearRect(0, 0, enhancedCanvas.width, enhancedCanvas.height);
    }
});

meshBtn.addEventListener('click', () => {
    showMesh = !showMesh;
    meshBtn.classList.toggle('active', showMesh);
    if (!showMesh) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
});

const cameraBtn = document.getElementById('cameraBtn');

cameraBtn.addEventListener('click', () => {
    showCamera = !showCamera;
    cameraBtn.classList.toggle('active', showCamera);
    videoElement.style.display = showCamera ? 'block' : 'none';
    overlayCanvas.style.display = showCamera ? 'block' : 'none';
    enhancedCanvas.style.display = showCamera ? 'block' : 'none';
    if (!showCamera && !showMesh) {
        overlayCanvas.style.display = 'none';
    }
    if (!showCamera && !showEnhancement) {
        enhancedCanvas.style.display = 'none';
    }
});

bufferBtn.addEventListener('click', () => {
    const currentIndex = BUFFER_SIZES.indexOf(bufferSize);
    const nextIndex = (currentIndex + 1) % BUFFER_SIZES.length;
    bufferSize = BUFFER_SIZES[nextIndex];
    dataBuffer = [];
    times = [];
    bpm = 0;
    bpmHistory = [];
    signalSNR = 0;
    confidence = 0;
    bpmValueEl.style.color = 'var(--text-dark)';
    statusTextEl.style.color = 'var(--text-dark)';
    bufferDurationEl.textContent = Math.round(bufferSize / 30) + 's';
    updateUI(true);
});

let flashOn = false;
let videoTrack = null;

flashBtn.addEventListener('click', async () => {
    if (!videoElement.srcObject) return;
    
    if (!videoTrack) {
        videoTrack = videoElement.srcObject.getVideoTracks()[0];
    }
    
    if (videoTrack) {
        const capabilities = videoTrack.getCapabilities();
        if (capabilities.torch) {
            flashOn = !flashOn;
            await videoTrack.applyConstraints({
                advanced: [{ torch: flashOn }]
            });
            flashBtn.classList.toggle('active', flashOn);
        }
    }
});

const micStartBtn = document.getElementById('micStartBtn');
const micStatus = document.getElementById('micStatus');
const micDurationSelect = document.getElementById('micDuration');
const micBpmValue = document.getElementById('micBpmValue');
const micResult = document.getElementById('micResult');
const micGraphs = document.getElementById('micGraphs');
const micRawCanvas = document.getElementById('micRawCanvas');
const micFilteredCanvas = document.getElementById('micFilteredCanvas');
const micFFTCanvas = document.getElementById('micFFTCanvas');
const micRawCtx = micRawCanvas ? micRawCanvas.getContext('2d') : null;
const micFilteredCtx = micFilteredCanvas ? micFilteredCanvas.getContext('2d') : null;
const micFFTCtx = micFFTCanvas ? micFFTCanvas.getContext('2d') : null;

if (micStartBtn) {
    let mediaRecorder;
    let audioChunks = [];
    
    function playBeep(frequency = 880, duration = 200) {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.5;
        
        oscillator.start();
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration / 1000);
        oscillator.stop(audioCtx.currentTime + duration / 1000);
    }
    
    function calculateBPMFromAudio(audioBuffer) {
        try {
            const data = audioBuffer.getChannelData(0);
            const sampleRate = audioBuffer.sampleRate;
            
            const step = Math.max(1, Math.floor(data.length / 500));
            const samples = [];
            for (let i = 0; i < data.length; i += step) {
                samples.push(data[i]);
            }
            
            const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
            const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;
            const threshold = mean + Math.sqrt(variance) * 1.5;
            
            const peaks = [];
            for (let i = 1; i < samples.length - 1; i++) {
                if (samples[i] > threshold && samples[i] > samples[i-1] && samples[i] > samples[i+1]) {
                    peaks.push(i);
                }
            }
            
            if (peaks.length < 2) return 0;
            
            const intervals = [];
            for (let i = 1; i < peaks.length; i++) {
                const interval = (peaks[i] - peaks[i-1]) * step / sampleRate;
                if (interval > 0.3 && interval < 2) {
                    intervals.push(interval);
                }
            }
            
            if (intervals.length === 0) return 0;
            
            const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
            return Math.round(60 / avgInterval);
        } catch (e) {
            console.error('BPM calculation error:', e);
            return 0;
        }
    }
    
function drawMicGraphs(audioBuffer, rawCtx) {
        try {
            const data = audioBuffer.getChannelData(0);
            
            const rawCanvas = rawCtx.canvas;
            
            rawCanvas.width = rawCanvas.offsetWidth * 2;
            rawCanvas.height = rawCanvas.offsetHeight * 2;
            
            const colors = getGraphColors();
            const step = Math.max(1, Math.floor(data.length / rawCanvas.width));
            
            rawCtx.fillStyle = colors.bg;
            rawCtx.fillRect(0, 0, rawCanvas.width, rawCanvas.height);
            
            rawCtx.strokeStyle = colors.line;
            rawCtx.lineWidth = 1;
            for (let y = 0; y < rawCanvas.height; y += rawCanvas.height / 4) {
                rawCtx.beginPath();
                rawCtx.moveTo(0, y);
                rawCtx.lineTo(rawCanvas.width, y);
                rawCtx.stroke();
            }
            
            rawCtx.strokeStyle = colors.primary;
            rawCtx.lineWidth = 2;
            rawCtx.beginPath();
            for (let i = 0; i < rawCanvas.width; i++) {
                const dataIdx = Math.min(Math.floor(i * step), data.length - 1);
                const y = rawCanvas.height / 2 + data[dataIdx] * rawCanvas.height * 0.4;
                if (i === 0) rawCtx.moveTo(i, y);
                else rawCtx.lineTo(i, y);
            }
            rawCtx.stroke();
            
            rawCtx.fillStyle = colors.primary;
            rawCtx.font = `${Math.round(rawCanvas.width / 30)}px Inter`;
            rawCtx.textAlign = 'left';
            rawCtx.fillText('Raw Audio', 15, 30);
        } catch (e) {
            console.error('Error drawing mic graph:', e);
        }
    }
    
    function applyBandpassFilter(data, sampleRate, lowFreq, highFreq) {
        const filtered = new Float32Array(data.length);
        let prev = 0;
        let prevInput = 0;
        
        for (let i = 0; i < data.length; i++) {
            filtered[i] = 0.95 * (prev + data[i] - prevInput);
            prev = filtered[i];
            prevInput = data[i];
        }
        
        return filtered;
    }
    
    function computeFFT(data, sampleRate) {
        const n = Math.min(4096, data.length);
        const step = Math.floor(data.length / n);
        
        const window = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
        }
        
        const magnitudes = new Float32Array(n);
        
        for (let k = 0; k < n; k++) {
            let real = 0, imag = 0;
            for (let j = 0; j < n; j++) {
                const angle = -2 * Math.PI * k * j / n;
                real += data[j * step] * window[j] * Math.cos(angle);
                imag += data[j * step] * window[j] * Math.sin(angle);
            }
            magnitudes[k] = Math.sqrt(real * real + imag * imag);
        }
        
        return magnitudes;
    }
    
    micStartBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const duration = parseInt(micDurationSelect.value) * 1000;
        
        micStatus.textContent = 'Move phone to chest...';
        micStartBtn.disabled = true;
        
        let countdown = 5;
        const countdownInterval = setInterval(() => {
            countdown--;
            micStatus.textContent = `Move to chest in ${countdown}...`;
            if (countdown <= 0) {
                clearInterval(countdownInterval);
            }
        }, 1000);
        
        setTimeout(() => {
            clearInterval(countdownInterval);
            playBeep(880, 300);
            micStatus.textContent = 'Recording... Keep quiet!';
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                playBeep(440, 300);
                micStatus.textContent = 'Processing...';
                
                await new Promise(resolve => setTimeout(resolve, 100));
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                
                const bpm = calculateBPMFromAudio(audioBuffer);
                
                micBpmValue.textContent = bpm > 0 ? bpm : '--';
                micStatus.textContent = bpm > 0 ? 'Done!' : 'Could not detect heartbeat';
                micResult.classList.remove('hidden');
                
                if (micGraphs) micGraphs.classList.remove('hidden');
                if (micRawCtx) {
                    await new Promise(resolve => requestAnimationFrame(resolve));
                    drawMicGraphs(audioBuffer, micRawCtx);
                }
                
                micStartBtn.disabled = false;
                
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            
            setTimeout(() => {
                mediaRecorder.stop();
            }, duration);
            
        }, 5000);
        
    } catch (err) {
        micStatus.textContent = 'Microphone access denied';
        micStartBtn.disabled = false;
    }
});
}