# Pulse Monitor

A real-time heart rate monitoring application that uses a webcam and machine learning to measure pulse from facial blood flow. No sensors or wearables needed - just a camera and a face.

<img width="1007" height="897" alt="image" src="https://github.com/user-attachments/assets/89a49c45-7f0b-48a8-81f2-52fe72e2dc88" />
<em>Guy in the photo: Andrew Horvitz (https://www.youtube.com/watch?v=aF4LG75-dZ8)</em>

## Running the Project

Open `index.html` in a modern browser (Chrome, Firefox, or Edge). Camera access will be requested when starting.

Or try it live without downloading: [https://chris-dev.at/projects/pulse-monitor/](https://chris-dev.at/projects/pulse-monitor/)

Click "Start Monitoring" and wait approximately 10 seconds for the app to collect enough data. Around 250 frames (~8 seconds at 30fps) are needed before a BPM can be calculated.

## Where the Idea Came From

This technique was discovered through Steve Mould's YouTube video "[Reveal Invisible Motion With This Clever Video Trick](https://www.youtube.com/watch?v=rEoc0YoALt0)". He demonstrates how tiny movements in video can be amplified to reveal things the naked eye cannot see. This project builds on that concept to create a working heart rate monitor.

## How It Works

The app is based on a simple physiological fact: when the heart beats, blood flows through the face, causing subtle changes in skin color - specifically in the green channel. These changes are invisible to the naked eye but can be detected by analyzing video frames.

### Step by Step

**1. Face Detection**
The app uses Google's MediaPipe Face Mesh to track facial landmarks in real-time. It identifies approximately 468 points on the face, and 10 key points around the forehead and cheek area are selected as the Region of Interest (ROI).

**2. Signal Extraction**
For each video frame, the pixels inside the ROI are analyzed and the average green value is calculated. Over time, this builds a signal that represents how blood flow changes in the face.

**3. Signal Processing**
Raw webcam data contains noise from various sources - lighting changes, camera noise, and head movement. The signal is cleaned up as follows:
- **Detrending** removes slow drifts caused by gradual movements
- **Normalization** scales the data to a standard range for consistent comparison
- **Bandpass filtering** isolates frequencies between 0.75 and 3.0 Hz - the range where the pulse signal lives (roughly 45 to 180 BPM)

**4. Finding the Heart Rate**
FFT (Fast Fourier Transform) converts the time-domain signal into frequency data. The highest peak in the 0.75-3.0 Hz range indicates the heart rate in beats per second. Multiplying by 60 gives the final BPM value.

**5. Smoothing**
Individual BPM calculations can fluctuate, so a rolling average of the last 15 values maintains a stable reading.


### Tips for Better Results

- Consistent, good lighting is important - avoid strong backlighting or moving between light and dark areas
- Keep the face relatively still during measurement
- Maintain a distance of about an arm's length from the camera
- The forehead region typically produces the cleanest signal


## Technical Details

- **Face Detection**: MediaPipe Face Mesh (loaded via CDN)
- **Signal Buffer**: 250 frames
- **Processing Rate**: ~30 fps
- **Bandpass Filter**: 0.75 - 3.0 Hz
- **BPM History**: Last 15 values averaged

The app runs entirely in the browser - no server required, no data leaves the local machine.

## Browser Requirements

- WebGL support
- getUserMedia API access
- Modern JavaScript (ES6+)

Works on desktop and mobile browsers, though performance is best on desktop with a dedicated webcam.

## License

MIT - do whatever you want.

---

And yes, like I'm going to write my own README. Of course AI wrote it - I'm going to leave that work to the professionals xD
