# AI-Based Traffic Monitoring System

Intermediate/advanced student project using Python 3.10, Streamlit, YOLO, OpenCV, and ByteTrack.

## Features
- Recorded video analysis and live browser webcam
- YOLO vehicle detection: car, motorcycle, bus, truck
- ByteTrack IDs for tracking across frames
- Virtual line-crossing counter with upward/downward direction counts (each track ID counted at most once)
- Low / Medium / High congestion estimate with configurable thresholds
- Simulated signal-priority recommendation
- CSV analytics download for recorded videos

## Requirements
- Windows 10/11
- Python 3.10 (64-bit recommended)
- VS Code
- Internet for first install and first YOLO model download
- Webcam for live camera mode (optional)

## Setup in Windows CMD
Open CMD in this project folder, then run:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3.10` is not recognized, try:
```bat
python --version
python -m venv .venv
.venv\Scripts\activate
```
Ensure the version shown is Python 3.10.x.

## Setup in VS Code
1. Extract the ZIP.
2. Open the extracted `AI_Traffic_Monitoring_System` folder in VS Code.
3. Open Terminal > New Terminal.
4. Run the setup commands above.
5. If prompted, select interpreter `.venv` in VS Code.
6. Run `streamlit run app.py`.
7. Open the local URL shown in terminal, normally `http://localhost:8501`.

## How to use
### Video Upload
1. Select a YOLO model in the sidebar.
2. Set detection confidence and line position.
3. Upload an MP4/AVI/MOV/MKV file.
4. Click **Process video**.
5. View detection overlay, metrics, chart, and download CSV.

### Live Webcam
1. Open the Live Webcam tab.
2. Click **START** in the webcam widget.
3. Allow camera permission in the browser.
4. Click **STOP** to end the session.

Webcam access works on `localhost` or HTTPS in most browsers. Browser WebRTC permissions/firewall settings may affect access.

## How counting works
- YOLO detects vehicles in each frame.
- ByteTrack attempts to keep a track ID for each vehicle.
- The vehicle centre is compared between frames.
- When its centre crosses the horizontal virtual line, the track ID is counted once, with upward/downward direction totals.
- The same ID is not counted a second time during that analyzer session.

This is a useful demo counter, not a certified traffic count. Track IDs can switch or reset if vehicles are occluded or leave/re-enter the scene. For bidirectional counting or multiple lanes, extend the line-crossing logic with direction-specific counters and per-lane regions.

## Congestion thresholds
The current estimate uses the number of vehicle detections in the current frame:
- Low: fewer than the low threshold
- Medium: from the low threshold up to (but not including) the high threshold
- High: at or above the high threshold

These values depend on camera placement and field of view. Tune them for each video.

## Troubleshooting
### `ModuleNotFoundError`
Activate the virtual environment and run:
```bat
python -m pip install -r requirements.txt
```

### Python version issue
```bat
python --version
where python
```

### Model download fails
Check internet access, then retry. The first run downloads `yolov8n.pt`. You may manually place a compatible model file in the project folder and select its filename in the app.

### Webcam does not start
- Allow camera permission for `localhost`.
- Close other apps that are using the webcam.
- Try a modern Chrome/Edge browser.
- Check Windows Settings > Privacy & security > Camera.

### Slow processing
Use `yolov8n.pt`, lower video resolution, or use a shorter video. CPU-only inference can be slow.

## Safety note
The signal-priority message is a simulated recommendation only. Do not connect this demo to physical traffic signals or use it for real traffic-control decisions.
