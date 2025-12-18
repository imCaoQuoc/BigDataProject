"""
Kafka Consumer Module
Handles consuming segmentation results from Kafka for Streamlit display.
"""

import json
from typing import Generator, Dict, Any, Optional, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import threading
from collections import deque
from datetime import datetime


class SegmentationConsumer:
    """
    Kafka consumer for receiving segmentation results in real-time.
    
    Designed to work with Streamlit's rerun mechanism for live updates.
    """
    
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        """
        Initialize the Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka broker address (e.g., "localhost:9092")
            topic: Kafka topic to consume from
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self._connected = False
        self._message_buffer: deque = deque(maxlen=1000)
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """
        Establish connection to Kafka broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000
            )
            self._connected = True
            print(f"✓ Kafka consumer connected to {self.bootstrap_servers}")
            return True
        except KafkaError as e:
            self._connected = False
            print(f"✗ Failed to connect to Kafka: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if consumer is connected to Kafka."""
        return self._connected and self.consumer is not None
    
    def consume_messages(self, timeout_ms: int = 1000, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Consume messages from Kafka topic.
        
        Non-blocking method suitable for Streamlit polling.
        
        Args:
            timeout_ms: Maximum time to wait for messages (milliseconds)
            max_messages: Maximum number of messages to return
            
        Returns:
            List of segmentation result dictionaries
        """
        if not self.is_connected:
            return []
        
        messages = []
        try:
            # Poll for messages
            records = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_messages)
            
            for topic_partition, record_list in records.items():
                for record in record_list:
                    messages.append(record.value)
                    self._message_buffer.append(record.value)
                    
        except KafkaError as e:
            print(f"✗ Error consuming messages: {e}")
        
        return messages
    
    def get_buffered_messages(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from internal buffer.
        
        Args:
            count: Number of recent messages to return (None = all)
            
        Returns:
            List of buffered messages
        """
        if count is None:
            return list(self._message_buffer)
        return list(self._message_buffer)[-count:]
    
    def clear_buffer(self) -> None:
        """Clear the internal message buffer."""
        self._message_buffer.clear()
    
    def start_background_consumer(self, callback=None) -> None:
        """
        Start consuming messages in a background thread.
        
        Args:
            callback: Optional function to call for each message
        """
        if self._running:
            return
        
        self._running = True
        
        def consume_loop():
            while self._running:
                try:
                    messages = self.consume_messages(timeout_ms=500)
                    if callback and messages:
                        for msg in messages:
                            callback(msg)
                except Exception as e:
                    print(f"Background consumer error: {e}")
        
        self._consumer_thread = threading.Thread(target=consume_loop, daemon=True)
        self._consumer_thread.start()
        print("✓ Background consumer started")
    
    def stop_background_consumer(self) -> None:
        """Stop the background consumer thread."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2)
            self._consumer_thread = None
        print("✓ Background consumer stopped")
    
    def seek_to_beginning(self) -> None:
        """Reset consumer to beginning of topic."""
        if self.consumer:
            self.consumer.seek_to_beginning()
    
    def seek_to_end(self) -> None:
        """Reset consumer to end of topic (latest messages only)."""
        if self.consumer:
            try:
                # Need to poll first to get partition assignment
                self.consumer.poll(timeout_ms=1000)
                self.consumer.seek_to_end()
                # Clear internal buffer to discard any old messages
                self._message_buffer.clear()
                print("✓ Consumer seeked to end - old messages cleared")
            except Exception as e:
                # Silently ignore - consumer will still work with auto_offset_reset='latest'
                print(f"⚠️ seek_to_end skipped (will use latest offset): {e}")
    
    def close(self) -> None:
        """Close the consumer connection."""
        self.stop_background_consumer()
        if self.consumer:
            self.consumer.close()
            self._connected = False
            print("✓ Kafka consumer closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class MessageFormatter:
    """
    Utility class to format Kafka messages for display.
    """
    
    @staticmethod
    def format_for_display(message: Dict[str, Any]) -> str:
        """
        Format a segmentation result message for log display.
        
        Args:
            message: Segmentation result dictionary
            
        Returns:
            Formatted string for display
        """
        if message.get("type") == "STATUS":
            status = message.get("status", "UNKNOWN")
            timestamp = MessageFormatter._format_timestamp(message.get("timestamp", ""))
            return f"[{timestamp}] 📢 Status: {status}"
        
        frame_idx = message.get("frame_index", "?")
        num_objects = message.get("num_objects", 0)
        proc_time = message.get("processing_time_ms", 0)
        model_type = message.get("model_type", "YOLO")
        timestamp = MessageFormatter._format_timestamp(message.get("timestamp", ""))
        
        lines = [f"[{timestamp}] 🖼️ {model_type} | Frame {frame_idx}: {num_objects} objects ({proc_time}ms)"]
        
        for det in message.get("detections", [])[:5]:  # Show max 5 detections
            class_name = det.get("class_name", "unknown")
            confidence = det.get("confidence", 0)
            bbox = det.get("bbox", [])
            bbox_str = f"[{', '.join(f'{int(x)}' for x in bbox)}]" if bbox else "[]"
            lines.append(f"  └─ {class_name} ({confidence:.2f}) at {bbox_str}")
        
        if len(message.get("detections", [])) > 5:
            lines.append(f"  └─ ... and {len(message['detections']) - 5} more")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_timestamp(iso_timestamp: str) -> str:
        """Format ISO timestamp to HH:MM:SS format."""
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%H:%M:%S")
        except:
            return "??:??:??"
    
    @staticmethod
    def format_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of multiple messages.
        
        Args:
            messages: List of segmentation result messages
            
        Returns:
            Summary dictionary with statistics
        """
        total_detections = 0
        class_counts = {}
        total_time = 0
        
        for msg in messages:
            if msg.get("type") == "STATUS":
                continue
            
            total_time += msg.get("processing_time_ms", 0)
            
            for det in msg.get("detections", []):
                total_detections += 1
                class_name = det.get("class_name", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "total_frames": len([m for m in messages if m.get("type") != "STATUS"]),
            "total_detections": total_detections,
            "class_counts": class_counts,
            "avg_processing_time_ms": total_time / len(messages) if messages else 0
        }
