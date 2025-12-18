"""
Video Processor Module
Handles video file processing using OpenCV to extract frames for segmentation.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import tempfile
import os


class VideoProcessor:
    """
    OpenCV-based video processor for extracting frames from video files.
    
    Attributes:
        frame_skip (int): Process every Nth frame for performance
        max_frames (int): Maximum number of frames to process (0 = unlimited)
    """
    
    def __init__(self, frame_skip: int = 5, max_frames: int = 100):
        """
        Initialize the video processor.
        
        Args:
            frame_skip: Process every Nth frame (default: 5)
            max_frames: Maximum frames to process, 0 for unlimited (default: 100)
        """
        self.frame_skip = max(1, frame_skip)
        self.max_frames = max_frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata:
                - fps: Frames per second
                - total_frames: Total number of frames
                - duration: Duration in seconds
                - width: Frame width
                - height: Frame height
                - codec: Video codec
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Convert fourcc to string
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            duration = total_frames / fps if fps > 0 else 0
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "duration": round(duration, 2),
                "width": width,
                "height": height,
                "codec": codec
            }
        finally:
            cap.release()
    
    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from a video file as a generator.
        
        Yields frames according to frame_skip setting for efficient processing.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple of (frame_index, frame_array) where:
                - frame_index: The original frame number in the video
                - frame_array: NumPy array of the frame (BGR format)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            frame_index = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if we should process this frame based on frame_skip
                if frame_index % self.frame_skip == 0:
                    yield (frame_index, frame)
                    processed_count += 1
                    
                    # Check max_frames limit
                    if self.max_frames > 0 and processed_count >= self.max_frames:
                        break
                
                frame_index += 1
        finally:
            cap.release()
    
    def extract_frames_from_bytes(self, video_bytes: bytes, 
                                   file_extension: str = ".mp4") -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video bytes (useful for Streamlit file uploads).
        
        Args:
            video_bytes: Raw video file bytes
            file_extension: File extension for temp file (default: .mp4)
            
        Yields:
            Tuple of (frame_index, frame_array)
        """
        # Create temporary file to save uploaded video
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Extract frames from temporary file
            yield from self.extract_frames(tmp_path)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_video_info_from_bytes(self, video_bytes: bytes, 
                                   file_extension: str = ".mp4") -> dict:
        """
        Get video info from bytes (useful for Streamlit file uploads).
        
        Args:
            video_bytes: Raw video file bytes
            file_extension: File extension for temp file (default: .mp4)
            
        Returns:
            Dictionary containing video metadata
        """
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            return self.get_video_info(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def count_frames_to_process(self, total_frames: int) -> int:
        """
        Calculate how many frames will actually be processed.
        
        Args:
            total_frames: Total frames in the video
            
        Returns:
            Number of frames that will be processed
        """
        frames_after_skip = (total_frames + self.frame_skip - 1) // self.frame_skip
        
        if self.max_frames > 0:
            return min(frames_after_skip, self.max_frames)
        return frames_after_skip
