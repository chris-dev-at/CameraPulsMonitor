const ROI_LANDMARKS = [10, 67, 103, 104, 108, 151, 337, 332, 297, 338];
const BUFFER_SIZE = 250;
const MAX_HISTORY = 15;

let faceMesh;
let camera;
let dataBuffer = [];
let times = [];
let bpm = 0;
let bpmHistory = [];
let isProcessing = false;
let lastFrameTime = 0;
let fftData = [];
let fftFreqs = [];
let peakFreq = 0;
let signalSNR = 0;
let showEnhancement = false;
let showMesh = true;
let enhancedImageData = null;
let lastLandmarks = null;

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

const bpmValueEl = document.getElementById('bpmValue');
const statusBadgeEl = document.getElementById('statusBadge');
const statusTextEl = document.getElementById('statusText');
const bufferProgressEl = document.getElementById('bufferProgress');
const qualityValueEl = document.getElementById('qualityValue');
const startBtn = document.getElementById('startBtn');
const mainApp = document.getElementById('mainApp');

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
    if (dataBuffer.length < BUFFER_SIZE) return;
    
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
        overlayCtx.fillStyle = '#ff006e';
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
        
        enhancedCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        enhancedCtx.fillRect(w - 190, 10, 180, 50);
        enhancedCtx.fillStyle = '#00d4ff';
        enhancedCtx.font = 'bold 14px Inter';
        enhancedCtx.fillText('Signal: ' + Math.round(normalizedSignal * 100) + '%', w - 180, 32);
        enhancedCtx.fillStyle = '#fff';
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
    
    graphCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    graphCtx.fillRect(0, 0, w, h);
    
    graphCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    graphCtx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {
        graphCtx.beginPath();
        graphCtx.moveTo(0, y);
        graphCtx.lineTo(w, y);
        graphCtx.stroke();
    }
    
    if (dataBuffer.length < 2) {
        graphCtx.fillStyle = '#8892b0';
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
    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 212, 255, 0.8)');
    
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
    
    graphCtx.strokeStyle = '#00d4ff';
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
    
    graphCtx.fillStyle = '#fff';
    graphCtx.font = `bold ${Math.round(w / 30)}px Inter`;
    graphCtx.textAlign = 'left';
    graphCtx.fillText('RAW SIGNAL', 15, 30);
}

function drawFFT() {
    const rect = fftCanvas.getBoundingClientRect();
    fftCanvas.width = rect.width * 2;
    fftCanvas.height = rect.height * 2;
    
    const w = fftCanvas.width;
    const h = fftCanvas.height;
    
    fftCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    fftCtx.fillRect(0, 0, w, h);
    
    fftCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    fftCtx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {
        fftCtx.beginPath();
        fftCtx.moveTo(0, y);
        fftCtx.lineTo(w, y);
        fftCtx.stroke();
    }
    
    if (fftData.length === 0 || fftFreqs.length === 0) {
        fftCtx.fillStyle = '#8892b0';
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
    gradient.addColorStop(0, 'rgba(123, 44, 191, 0.2)');
    gradient.addColorStop(1, 'rgba(123, 44, 191, 0.8)');
    
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
    
    fftCtx.strokeStyle = '#a855f7';
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
                
                fftCtx.shadowColor = '#ff006e';
                fftCtx.shadowBlur = 15;
                fftCtx.fillStyle = '#ff006e';
                fftCtx.beginPath();
                fftCtx.arc(px, py, 8, 0, 2 * Math.PI);
                fftCtx.fill();
                fftCtx.shadowBlur = 0;
            }
        }
    }
    
    fftCtx.fillStyle = '#fff';
    fftCtx.font = `bold ${Math.round(w / 30)}px Inter`;
    fftCtx.textAlign = 'left';
    fftCtx.fillText('FFT SPECTRUM', 15, 30);
}

function updateUI() {
    const progress = (dataBuffer.length / BUFFER_SIZE) * 100;
    
    if (dataBuffer.length >= BUFFER_SIZE) {
        bpmValueEl.textContent = Math.round(bpm);
        statusTextEl.textContent = 'Measuring...';
        statusBadgeEl.className = 'status-badge';
        
        const snrPercent = Math.min(100, Math.round((signalSNR / 3) * 100));
        qualityValueEl.textContent = snrPercent + '%';
    } else {
        bpmValueEl.textContent = '--';
        statusTextEl.textContent = `Collecting data... ${Math.round(progress)}%`;
        statusBadgeEl.className = 'status-badge warning';
        qualityValueEl.textContent = '--';
    }
    
    bufferProgressEl.textContent = Math.round(progress) + '%';
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
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        drawOverlay(landmarks);
        
        const greenVal = getROIAverage(landmarks);
        
        if (greenVal > 0) {
            dataBuffer.push(greenVal);
            times.push(performance.now() / 1000);
            
            if (dataBuffer.length > BUFFER_SIZE) {
                dataBuffer.shift();
                times.shift();
            }
            
            if (dataBuffer.length >= BUFFER_SIZE) {
                calculateBPM();
            }
        }
    } else {
        statusTextEl.textContent = 'No face detected';
        statusBadgeEl.className = 'status-badge error';
    }
    
    updateUI();
}

async function initFaceMesh() {
    try {
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
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 }, 
                facingMode: 'user' 
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

startBtn.addEventListener('click', async () => {
    console.log('Start button clicked');
    startBtn.disabled = true;
    startBtn.textContent = 'Loading AI Model...';
    
    try {
        await initFaceMesh();
        startBtn.classList.add('hidden');
        mainApp.classList.remove('hidden');
    } catch (err) {
        console.error('Error:', err);
        startBtn.disabled = false;
        startBtn.textContent = 'Error - Click to Retry';
        statusTextEl.textContent = 'Failed to load. Check console for details.';
        statusBadgeEl.className = 'status-badge error';
    }
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