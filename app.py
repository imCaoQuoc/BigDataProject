"""
Video Segmentation System - Streamlit Application
Main UI for video upload, YOLO segmentation, and real-time Kafka log streaming.
Supports processing multiple videos without page refresh.
"""

import streamlit as st
import time
import tempfile
import os
import cv2
import numpy as np
from datetime import datetime

# Import modules
from modules.video_processor import VideoProcessor
from modules.segmentation import create_segmenter, get_available_models
from modules.kafka_producer import SegmentationProducer
from modules.kafka_consumer import SegmentationConsumer, MessageFormatter
import config

# Page configuration
st.set_page_config(
    page_title="Video Segmentation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .status-box {
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-connected {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-disconnected {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .gpu-badge {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'processing': False,
        'stop_requested': False,
        'kafka_logs': [],
        'total_detections': 0,
        'frames_processed': 0,
        'class_counts': {},
        'consumer': None,
        'consumer_connected': False,
        'video_path': None,
        'last_annotated_frame': None,
        'last_video_name': None,
        'selected_model': 'yolo',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def connect_kafka_consumer():
    """Connect to Kafka consumer."""
    import time
    try:
        # Use unique group_id to ensure we start fresh (ignore old messages)
        unique_group_id = f"{config.KAFKA_GROUP_ID}-{int(time.time() * 1000)}"
        
        consumer = SegmentationConsumer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            topic=config.KAFKA_TOPIC,
            group_id=unique_group_id
        )
        if consumer.connect():
            st.session_state.consumer = consumer
            st.session_state.consumer_connected = True
            # Clear old Kafka logs and stats on new connection
            st.session_state.kafka_logs = []
            st.session_state.frames_processed = 0
            st.session_state.total_detections = 0
            st.session_state.class_counts = {}
            print(f"✓ Kafka consumer connected with group: {unique_group_id}")
            return True
    except Exception as e:
        st.error(f"Failed to connect Kafka consumer: {e}")
    return False


def disconnect_kafka_consumer():
    """Disconnect Kafka consumer."""
    if st.session_state.consumer:
        st.session_state.consumer.close()
        st.session_state.consumer = None
        st.session_state.consumer_connected = False


def poll_kafka_messages():
    """Poll for new Kafka messages and update logs."""
    if st.session_state.consumer and st.session_state.consumer_connected:
        try:
            messages = st.session_state.consumer.consume_messages(timeout_ms=100, max_messages=50)
            
            # Debug: Print when messages are received
            if messages:
                print(f"📥 Received {len(messages)} messages from Kafka")
            
            for msg in messages:
                formatted = MessageFormatter.format_for_display(msg)
                st.session_state.kafka_logs.append(formatted)
                
                if msg.get("type") != "STATUS":
                    st.session_state.frames_processed += 1
                    st.session_state.total_detections += msg.get("num_objects", 0)
                    
                    for det in msg.get("detections", []):
                        class_name = det.get("class_name", "unknown")
                        st.session_state.class_counts[class_name] = \
                            st.session_state.class_counts.get(class_name, 0) + 1
            
            if len(st.session_state.kafka_logs) > 500:
                st.session_state.kafka_logs = st.session_state.kafka_logs[-500:]
                
            return len(messages)
        except Exception as e:
            print(f"❌ Error in poll_kafka_messages: {e}")
    else:
        # Debug: Print if consumer not connected
        if not hasattr(st.session_state, '_poll_warning_shown'):
            print("⚠️ poll_kafka_messages called but consumer not connected!")
            st.session_state._poll_warning_shown = True
    return 0


def save_uploaded_video(uploaded_file):
    """Save uploaded video to a persistent temp file."""
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        
        # Clean up previous temp file
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            try:
                os.unlink(st.session_state.video_path)
            except:
                pass
        
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.video_path = tmp_file.name
        
        return st.session_state.video_path
    return None


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def process_video(video_path, video_name, frame_skip, max_frames, model_config,
                  frame_placeholder, progress_placeholder, status_placeholder,
                  stats_placeholder, log_placeholder):
    """Process video with real-time UI updates."""
    
    # Reset Kafka logs and stats for new processing
    st.session_state.kafka_logs = []
    st.session_state.total_detections = 0
    st.session_state.frames_processed = 0
    st.session_state.class_counts = {}
    st.session_state.last_video_name = video_name
    st.session_state._poll_warning_shown = False
    st.session_state.stop_requested = False  # Reset stop flag
    
    # Auto-reconnect Kafka consumer if not connected
    if not st.session_state.consumer_connected:
        status_placeholder.info("📡 Connecting Kafka consumer...")
        # Disconnect existing consumer if any
        if st.session_state.consumer:
            try:
                st.session_state.consumer.close()
            except:
                pass
            st.session_state.consumer = None
        # Reconnect
        connect_kafka_consumer()
        if st.session_state.consumer_connected:
            print("✓ Kafka consumer auto-reconnected for new model")
    
    try:
        status_placeholder.info("🔧 Initializing...")
        video_processor = VideoProcessor(frame_skip=frame_skip, max_frames=max_frames)
        
        model_type = model_config.get("type", "yolo")
        status_placeholder.info(f"🤖 Loading {model_type.upper()} model...")
        segmenter = create_segmenter(
            model_config=model_config,
            confidence=config.DEFAULT_CONFIDENCE,
            iou=config.DEFAULT_IOU
        )
        device_info = segmenter.get_device_info()
        
        status_placeholder.info("📡 Connecting to Kafka...")
        producer = SegmentationProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            topic=config.KAFKA_TOPIC
        )
        
        video_info = video_processor.get_video_info(video_path)
        total_frames = video_processor.count_frames_to_process(video_info['total_frames'])
        
        device_str = "🚀 GPU" if device_info['device'] == 'cuda' else "💻 CPU"
        status_placeholder.info(f"📹 {video_info['width']}x{video_info['height']} | {total_frames} frames | {device_str}")
        
        producer.send_status_message("START", {
            "video_name": video_name,
            "total_frames": total_frames,
            "device": device_info['device']
        })
        
        frames_done = 0
        last_ui_update = time.time()
        
        for frame_idx, frame in video_processor.extract_frames(video_path):
            # Check if stop was requested
            if st.session_state.stop_requested:
                status_placeholder.warning("⏹️ Processing stopped by user")
                break
            
            result, annotated_frame = segmenter.segment_frame_with_visualization(frame, frame_index=frame_idx)
            producer.send_result(result)
            
            frames_done += 1
            progress = frames_done / total_frames
            
            current_time = time.time()
            if current_time - last_ui_update > 0.1:
                progress_placeholder.progress(min(progress, 1.0), text=f"Frame {frame_idx} ({frames_done}/{total_frames})")
                
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_rgb, caption=f"Frame {frame_idx} - {result['num_objects']} objects")
                
                if st.session_state.consumer_connected:
                    poll_kafka_messages()
                    
                    with stats_placeholder.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🖼️ Frames", st.session_state.frames_processed)
                        c2.metric("🎯 Detections", st.session_state.total_detections)
                        c3.metric("📦 Classes", len(st.session_state.class_counts))
                    
                    if st.session_state.kafka_logs:
                        with log_placeholder.container(height=350):
                            log_text = "\n\n".join(reversed(st.session_state.kafka_logs[-20:]))
                            st.code(log_text, language=None)
                
                last_ui_update = current_time
        
        # Save last frame
        st.session_state.last_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        producer.send_status_message("END", {"total_processed": frames_done})
        producer.flush()
        producer.close()
        
        progress_placeholder.progress(1.0, text="✅ Complete!")
        status_placeholder.success(f"✅ Processed {frames_done} frames! Upload a new video to continue.")
        
        # Final updates
        if st.session_state.consumer_connected:
            time.sleep(0.3)
            poll_kafka_messages()
            with stats_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("🖼️ Frames", st.session_state.frames_processed)
                c2.metric("🎯 Detections", st.session_state.total_detections)
                c3.metric("📦 Classes", len(st.session_state.class_counts))
            if st.session_state.kafka_logs:
                with log_placeholder.container(height=350):
                    log_text = "\n\n".join(reversed(st.session_state.kafka_logs[-25:]))
                    st.code(log_text, language=None)
        
        return True
        
    except Exception as e:
        status_placeholder.error(f"❌ Error: {e}")
        return False


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("## 📁 Video Input")
        
        # File uploader - key changes when video is processed to allow new upload
        uploaded_file = st.file_uploader(
            "Upload video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="MP4, AVI, MOV, MKV, WebM",
            disabled=st.session_state.processing
        )
        
        video_path = None
        video_name = None
        if uploaded_file is not None:
            video_path = save_uploaded_video(uploaded_file)
            video_name = uploaded_file.name
            if video_path and os.path.exists(video_path):
                st.video(video_path)
        
        st.markdown("---")
        st.markdown("## 🤖 Model Selection")
        
        # Get available models and create selection options
        available_models = get_available_models(config.MODELS)
        model_options = list(available_models.keys())
        model_labels = {k: f"{v['description']} {'✅' if v.get('available', False) else '⚠️'}" 
                        for k, v in available_models.items()}
        
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            format_func=lambda x: model_labels.get(x, x),
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="Choose the segmentation model to use",
            disabled=st.session_state.processing
        )
        st.session_state.selected_model = selected_model
        
        # Show model details
        selected_config = config.MODELS.get(selected_model, {})
        if not available_models.get(selected_model, {}).get('available', False):
            st.warning(f"⚠️ Model checkpoint not found at: {selected_config.get('path', 'N/A')}")
        else:
            st.success(f"✅ Model ready")
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        frame_skip = st.slider(
            "Frame Skip", min_value=0, max_value=30, value=config.FRAME_SKIP,
            help="Process every Nth frame", disabled=st.session_state.processing
        )
        
        max_frames = st.slider(
            "Max Frames", min_value=0, max_value=500, value=config.MAX_FRAMES,
            help="Maximum frames (0 = unlimited)", disabled=st.session_state.processing
        )
        
        st.markdown("---")
        
        # Start and Stop buttons side by side
        model_available = available_models.get(selected_model, {}).get('available', False)
        can_start = uploaded_file is not None and not st.session_state.processing and model_available
        
        col_start, col_stop = st.columns(2)
        with col_start:
            start_clicked = st.button("▶️ Start", type="primary", disabled=not can_start, use_container_width=True)
        with col_stop:
            # Stop is always enabled (like F5) - instant reset
            if st.button("⏹️ Stop", type="secondary", use_container_width=True):
                # Reset all processing state immediately
                st.session_state.processing = False
                st.session_state.stop_requested = True
                st.session_state.kafka_logs = []
                st.session_state.frames_processed = 0
                st.session_state.total_detections = 0
                st.session_state.class_counts = {}
                st.session_state.last_annotated_frame = None
                # Instant restart like F5
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📡 Kafka")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", disabled=st.session_state.consumer_connected):
                connect_kafka_consumer()
                st.rerun()
        with col2:
            if st.button("⏹️ Disconnect", disabled=not st.session_state.consumer_connected):
                disconnect_kafka_consumer()
                st.rerun()
        
        if st.session_state.consumer_connected:
            st.success("✅ Connected")
        else:
            st.warning("❌ Disconnected")
        
        st.markdown("---")
        st.markdown("### 📊 Device")
        if check_gpu():
            st.markdown('<span class="gpu-badge">🚀 GPU Available</span>', unsafe_allow_html=True)
        else:
            st.caption("💻 CPU Only")
    
    # ==================== MAIN CONTENT ====================
    
    # Row 1: Headers
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### 🎯 Segmentation Masks")
    with col_right:
        st.markdown("### 📡 Kafka Logs")
    
    # Row 2: Metrics
    col_left2, col_right2 = st.columns([1, 1])
    with col_left2:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
    with col_right2:
        stats_placeholder = st.empty()
        with stats_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("🖼️ Frames", st.session_state.frames_processed)
            c2.metric("🎯 Detections", st.session_state.total_detections)
            c3.metric("📦 Classes", len(st.session_state.class_counts))
    
    # Row 3: Main content
    col_left3, col_right3 = st.columns([1, 1])
    with col_left3:
        frame_placeholder = st.empty()
        if st.session_state.last_annotated_frame is not None and not st.session_state.processing:
            with frame_placeholder.container(height=400):
                st.image(st.session_state.last_annotated_frame, caption=f"Last: {st.session_state.last_video_name}")
        elif not st.session_state.processing:
            with frame_placeholder.container(height=400):
                st.info("🖼️ Upload a video and click **Start Processing**")
    
    with col_right3:
        log_placeholder = st.empty()
        if st.session_state.kafka_logs and not st.session_state.processing:
            with log_placeholder.container(height=400):
                log_text = "\n\n".join(reversed(st.session_state.kafka_logs[-25:]))
                st.code(log_text, language=None)
        elif not st.session_state.processing:
            with log_placeholder.container(height=400):
                st.info("📭 Connect to Kafka and process video")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Refresh"):
                        st.rerun()
                with col_b:
                    if st.button("🗑️ Clear All"):
                        st.session_state.kafka_logs = []
                        st.session_state.total_detections = 0
                        st.session_state.frames_processed = 0
                        st.session_state.class_counts = {}
                        st.session_state.last_annotated_frame = None
                        st.rerun()
    
    # ==================== PROCESSING ====================
    if start_clicked and video_path:
        st.session_state.processing = True
        
        # Clear placeholders for processing
        frame_placeholder.empty()
        with frame_placeholder.container(height=400):
            st.info("⏳ Starting...")
        
        log_placeholder.empty()
        with log_placeholder.container(height=400):
            st.info("📭 Waiting for results...")
        
        # Run processing with selected model
        model_config = config.MODELS.get(st.session_state.selected_model, config.MODELS['yolo'])
        success = process_video(
            video_path=video_path,
            video_name=video_name,
            frame_skip=frame_skip,
            max_frames=max_frames,
            model_config=model_config,
            frame_placeholder=frame_placeholder,
            progress_placeholder=progress_placeholder,
            status_placeholder=status_placeholder,
            stats_placeholder=stats_placeholder,
            log_placeholder=log_placeholder
        )
        
        st.session_state.processing = False
        
        # Show final frame
        if success and st.session_state.last_annotated_frame is not None:
            with frame_placeholder.container(height=400):
                st.image(st.session_state.last_annotated_frame, caption=f"Completed: {video_name}")
    
    # Auto-refresh when consumer is connected (but not during processing)
    if st.session_state.consumer_connected and not st.session_state.processing:
        poll_kafka_messages()
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
