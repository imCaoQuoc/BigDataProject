"""
Kafka Producer Module
Handles sending segmentation results to Kafka for real-time streaming.
"""

import json
import numpy as np
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_safe_serializer(v):
    """Serialize dict to JSON with numpy type support."""
    return json.dumps(v, cls=NumpyEncoder).encode('utf-8')


class SegmentationProducer:
    """
    Kafka producer for streaming segmentation results.
    
    Serializes detection results to JSON and sends them to a Kafka topic.
    """
    
    def __init__(self, bootstrap_servers: str, topic: str):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker address (e.g., "localhost:9092")
            topic: Kafka topic to send messages to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._connected = False
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=numpy_safe_serializer,
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_block_ms=5000
            )
            self._connected = True
            print(f"✓ Kafka producer connected to {self.bootstrap_servers}")
        except KafkaError as e:
            self._connected = False
            print(f"✗ Failed to connect to Kafka: {e}")
            raise ConnectionError(f"Cannot connect to Kafka at {self.bootstrap_servers}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if producer is connected to Kafka."""
        return self._connected and self.producer is not None
    
    def send_result(self, result: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Send a segmentation result to Kafka.
        
        Args:
            result: Segmentation result dictionary from YOLOSegmenter
            key: Optional message key for partitioning
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.is_connected:
            print("✗ Producer not connected to Kafka")
            return False
        
        try:
            # Generate key based on frame index if not provided
            if key is None:
                key = f"frame_{result.get('frame_index', 0)}"
            
            model_type = result.get('model_type', 'UNKNOWN')
            frame_idx = result.get('frame_index', 0)
            
            # Send message asynchronously
            future = self.producer.send(
                self.topic,
                key=key,
                value=result
            )
            
            # Wait for confirmation (with timeout)
            future.get(timeout=10)
            
            # Debug: Print every 10th frame to avoid spam
            if frame_idx % 10 == 0:
                print(f"📤 [{model_type}] Frame {frame_idx} sent to Kafka")
            
            return True
            
        except KafkaError as e:
            print(f"✗ Failed to send message (Kafka): {e}")
            return False
        except Exception as e:
            # Catch JSON serialization errors and other exceptions
            model_type = result.get('model_type', 'UNKNOWN')
            print(f"✗ Failed to send message ({model_type}): {type(e).__name__}: {e}")
            return False
    
    def send_batch(self, results: list, flush_after: bool = True) -> int:
        """
        Send multiple results in batch.
        
        Args:
            results: List of segmentation result dictionaries
            flush_after: Whether to flush after sending batch
            
        Returns:
            Number of successfully sent messages
        """
        if not self.is_connected:
            return 0
        
        sent_count = 0
        for result in results:
            if self.send_result(result):
                sent_count += 1
        
        if flush_after:
            self.flush()
        
        return sent_count
    
    def send_status_message(self, status: str, details: Optional[Dict] = None) -> bool:
        """
        Send a status/control message to Kafka.
        
        Useful for signaling start/end of processing, errors, etc.
        
        Args:
            status: Status type (e.g., "START", "END", "ERROR")
            details: Optional additional details
            
        Returns:
            True if sent successfully
        """
        message = {
            "type": "STATUS",
            "status": status,
            "details": details or {},
            "timestamp": self._get_timestamp()
        }
        return self.send_result(message, key=f"status_{status.lower()}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def flush(self, timeout: float = 10.0) -> None:
        """
        Flush pending messages to Kafka.
        
        Args:
            timeout: Maximum time to wait for flush (seconds)
        """
        if self.producer:
            self.producer.flush(timeout=timeout)
    
    def close(self) -> None:
        """Close the producer connection."""
        if self.producer:
            self.producer.flush(timeout=5)
            self.producer.close()
            self._connected = False
            print("✓ Kafka producer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
