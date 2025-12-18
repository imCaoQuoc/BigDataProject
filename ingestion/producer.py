"""
Fire Video Frame Producer - Stream frames to Kafka
Supports multiple offline video sources.
"""

import cv2
import json
import time
import base64
import sys
import os
import glob
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fire-frames")
VIDEO_DIR = os.getenv("VIDEO_DIR", "/opt/spark/video") # Default mount point in Spark Worker
FRAME_RATE = int(os.getenv("FRAME_RATE", "30")) # Target FPS to process/send
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "640"))

def create_kafka_producer():
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                max_request_size=10485760, # 10MB
                buffer_memory=33554432,    # 32MB
                compression_type='gzip'
            )
            print(f"[Producer] ✅ Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except KafkaError as e:
            retry_count += 1
            print(f"[Producer] ⚠️ Connection attempt {retry_count}/{max_retries}: {e}")
            if retry_count < max_retries:
                time.sleep(5)
            else:
                raise

def process_video(video_path, producer):
    camera_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[Producer] Processing: {video_path} as Camera: {camera_id}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Producer] ❌ Cannot open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: fps = 30 # Fallback
    
    # Calculate frame skipping to match target FRAME_RATE
    # If video is 30fps and functionality is 30fps, skip=1.
    # If video is 60fps and target is 10fps, skip=6.
    # Process ALL frames, no skipping
    frame_skip = 1 
    
    frame_count = 0
    sent_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
                
            frame_count += 1
            # Resize
            height, width = frame.shape[:2]
            new_height = int(height * (RESIZE_WIDTH / width))
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, new_height))
            
            # Encode
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            message = {
                "camera_id": camera_id,
                "frame_number": sent_count,
                "timestamp": time.time(),
                "frame": frame_base64,
                "width": RESIZE_WIDTH,
                "height": new_height
            }
            
            producer.send(KAFKA_TOPIC, value=message)
            sent_count += 1
            
            # No sleep - BLAST MODE for full processing
            
            if sent_count % 100 == 0:
                print(f"[Producer] Camera {camera_id}: Sent {sent_count} frames")

    except KeyboardInterrupt:
        print("[Producer] Stopped by user")
        sys.exit(0)
    finally:
        cap.release()
        print(f"[Producer] Finished {video_path}. Total frames sent: {sent_count}")

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("🔥 Starting Fire Detection Video Producer")
    print("=" * 60)

    try:
        producer = create_kafka_producer()
        
        # Priority: test.mp4
        test_video = os.path.join(VIDEO_DIR, "test.mp4")
        if os.path.exists(test_video):
            video_files = [test_video]
            print(f"[Producer] 🎯 Target found: {test_video}")
        else:
            # Fallback
            search_pattern = os.path.join(VIDEO_DIR, "*")
            video_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"[Producer] ⚠️ No video files found in {VIDEO_DIR}")
            sys.exit(1)
            
        print(f"[Producer] Processing {len(video_files)} videos...")
        
        for video_file in video_files:
            # Process full speed, no sleep
            process_video(video_file, producer)

        producer.flush()
        producer.close()
        print("[Producer] ✅ All videos processed.")
        
    except Exception as e:
        print(f"[Producer] ❌ Fatal Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
