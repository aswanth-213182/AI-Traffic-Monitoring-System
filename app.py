import time
import tempfile
import os
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from traffic_engine import (
    TrafficAnalyzer, VEHICLE_NAMES, draw_dashboard_overlay,
    summarize_stats, congestion_level, signal_recommendation
)

st.set_page_config(page_title="AI Traffic Monitoring", page_icon="🚦", layout="wide")

st.title("🚦 AI-Based Traffic Monitoring System")
st.caption("YOLO vehicle detection • ID tracking • line-crossing counts • congestion analytics • simulated signal recommendations")

with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], index=0,
                                help="The model downloads automatically on first use.")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    line_position = st.slider("Virtual count-line position (%)", 10, 90, 55, 5)
    low_limit = st.number_input("Low congestion: max vehicles in frame", min_value=1, max_value=100, value=8)
    high_limit = st.number_input("High congestion: from vehicles in frame", min_value=2, max_value=200, value=18)
    if high_limit <= low_limit:
        st.warning("High congestion threshold should be greater than the low threshold.")

    st.markdown("---")
    st.markdown("**Vehicle classes**")
    st.write(", ".join(VEHICLE_NAMES.values()))
    st.caption("Signal output is a recommendation/simulation only. This app does not control real traffic lights.")

tab_video, tab_camera, tab_about = st.tabs(["🎞️ Video Upload", "📷 Live Webcam", "ℹ️ How it works"])

def show_metrics(stats):
    cols = st.columns(5)
    cols[0].metric("Vehicles in current frame", stats.get("current_vehicles", 0))
    cols[1].metric("Unique IDs observed", stats.get("unique_ids", 0))
    cols[2].metric("Line crossings", stats.get("line_crossings", 0))
    cols[3].metric("Congestion", stats.get("congestion", "Unknown"))
    cols[4].metric("FPS (approx.)", stats.get("fps", 0))

def display_analytics(rows, final_stats):
    st.subheader("📊 Traffic analytics")
    show_metrics(final_stats)
    if rows:
        df = pd.DataFrame(rows)
        if "timestamp_sec" in df.columns:
            counts = df.groupby(df["timestamp_sec"].astype(int), as_index=False).agg(
                vehicles=("current_vehicles", "max"),
                crossings=("line_crossings", "max")
            )
            fig = px.line(counts, x="timestamp_sec", y=["vehicles", "crossings"],
                          labels={"timestamp_sec": "Video time (seconds)", "value": "Count", "variable": "Metric"},
                          title="Vehicle presence and cumulative line crossings")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(20), use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download analytics CSV", csv, "traffic_analytics.csv", "text/csv")
    st.info(signal_recommendation(final_stats.get("current_vehicles", 0)))
    st.caption("Congestion is estimated from the number of detected vehicles in the current frame. Calibrate thresholds for your camera angle and location.")

with tab_video:
    st.subheader("Analyze a recorded traffic video")
    uploaded = st.file_uploader("Upload MP4, AVI, MOV, or MKV", type=["mp4", "avi", "mov", "mkv"])
    process_video = st.button("▶️ Process video", type="primary", disabled=uploaded is None)
    if process_video and uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open this video. Try a standard MP4 file.")
        else:
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_slot = st.empty()
            status_slot = st.empty()
            progress = st.progress(0)
            analyzer = TrafficAnalyzer(model_choice, confidence, line_position / 100.0)
            rows = []
            frame_index = 0
            start = time.time()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                processed, stats = analyzer.process_frame(frame, low_limit, high_limit)
                frame_index += 1
                elapsed = frame_index / fps_in if fps_in > 0 else 0
                stats["timestamp_sec"] = round(elapsed, 2)
                rows.append(stats.copy())
                frame_slot.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                status_slot.write(f"Processing frame {frame_index}" + (f" / {total_frames}" if total_frames else ""))
                if total_frames:
                    progress.progress(min(frame_index / total_frames, 1.0))
            cap.release()
            try:
                os_path = Path(video_path)
                os_path.unlink(missing_ok=True)
            except Exception:
                pass
            final_stats = analyzer.last_stats.copy()
            final_stats["fps"] = round(frame_index / max(time.time() - start, 0.001), 1)
            st.success(f"Finished processing {frame_index} frames.")
            display_analytics(rows, final_stats)

with tab_camera:
    st.subheader("Live webcam monitoring")
    st.write("Allow webcam access in your browser. For best results, point the camera at a visible traffic video or a safe, permitted view of a road.")
    if "camera_analyzer" not in st.session_state:
        st.session_state.camera_analyzer = None

    class WebcamProcessor:
        def __init__(self):
            self.analyzer = TrafficAnalyzer(model_choice, confidence, line_position / 100.0)
            self.low_limit = int(low_limit)
            self.high_limit = int(high_limit)
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed, stats = self.analyzer.process_frame(img, self.low_limit, self.high_limit)
            self.analyzer.last_stats = stats
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

    # Streamlit WebRTC callbacks run in a separate thread.
    import av
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="traffic-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=WebcamProcessor,
        async_processing=True,
    )
    st.caption("Live browser camera access requires HTTPS or localhost. The model may take a few seconds to load on first run.")
    if ctx.video_processor:
        stats = ctx.video_processor.analyzer.last_stats
        show_metrics(stats)
        st.info(signal_recommendation(stats.get("current_vehicles", 0)))
        st.caption("For live mode, metrics update as frames are processed. Use the Video Upload tab for downloadable frame-by-frame analytics.")

with tab_about:
    st.subheader("System workflow")
    st.markdown("""
    1. **Input:** upload a recorded video or enable the browser webcam.
    2. **Detection:** YOLO identifies supported vehicle classes (car, motorcycle, bus, truck).
    3. **Tracking:** ByteTrack associates detections with temporary track IDs across frames.
    4. **Counting:** a vehicle is counted once when its tracked centre crosses the virtual line.
    5. **Congestion estimate:** current detected vehicle count is mapped to configurable Low / Medium / High thresholds.
    6. **Recommendation:** a simple simulated signal-priority message is generated from estimated density.

    **Limitations:** results depend on camera angle, lighting, occlusion, frame rate, and model quality. Track IDs can occasionally switch, so line-crossing counts are estimates, not certified traffic measurements.
    """)
