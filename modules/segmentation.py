"""
Multi-Architecture Segmentation Module
Supports YOLO, UNet, DeepLabV3+, and SegFormer models with GPU support.
"""

import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import os


class BaseSegmenter:
    """Base class for all segmenters with common interface."""
    
    def __init__(self, model_config: Dict[str, Any], confidence: float = 0.25, device: str = None):
        self.model_config = model_config
        self.model_path = model_config.get("path", "")
        self.confidence = confidence
        self.img_size = model_config.get("img_size", 512)
        self.model = None
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the compute device."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        return info
    
    def segment_frame_with_visualization(self, frame: np.ndarray, 
                                          frame_index: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
        """Run segmentation and return both results and annotated frame."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_available(self) -> bool:
        """Check if the model file/directory exists."""
        return os.path.exists(self.model_path)


class YOLOSegmenter(BaseSegmenter):
    """
    YOLO-based instance segmentation wrapper.
    Uses Ultralytics YOLO models.
    """
    
    def __init__(self, model_config: Dict[str, Any], confidence: float = 0.25, 
                 iou: float = 0.45, device: str = None):
        super().__init__(model_config, confidence, device)
        self.iou = iou
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model from the specified path."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            if self.device == 'cuda':
                self.model.to('cuda')
            print(f"✓ YOLO model loaded: {self.model_path}")
            print(f"✓ Using device: {self.device.upper()}" + 
                  (f" ({torch.cuda.get_device_name(0)})" if self.device == 'cuda' else ""))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the class names from the loaded model."""
        if self.model is None:
            return {}
        return self.model.names
    
    def segment_frame_with_visualization(self, frame: np.ndarray, 
                                          frame_index: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
        """Run segmentation and return both results and annotated frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = datetime.now()
        
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Custom green overlay instead of default plot()
        annotated_frame = self._create_green_overlay(frame, results[0])
        detections = self._process_results(results[0])
        
        result_dict = {
            "frame_index": int(frame_index),
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "num_objects": int(len(detections)),
            "processing_time_ms": round(float(processing_time), 2),
            "device": str(self.device),
            "model_type": "YOLO"
        }
        
        return result_dict, annotated_frame
    
    def _create_green_overlay(self, frame: np.ndarray, result) -> np.ndarray:
        """Create green overlay for YOLO results to match other models."""
        overlay = frame.copy()
        color = (0, 255, 0)  # BGR - Green
        alpha = 0.4
        
        boxes = result.boxes
        masks = result.masks
        
        if boxes is None or len(boxes) == 0:
            return overlay
        
        for i in range(len(boxes)):
            # Draw mask if available
            if masks is not None and i < len(masks.data):
                mask = masks.data[i].cpu().numpy()
                # Resize mask to frame size
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bool = mask > 0.5
                if mask_bool.any():
                    overlay[mask_bool] = (
                        overlay[mask_bool] * (1 - alpha) + 
                        np.array(color) * alpha
                    ).astype(np.uint8)
            
            # Draw bounding box
            bbox = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = self.model.names.get(class_id, "fire")
            label = f"{class_name} {confidence*100:.0f}%"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(overlay, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)
        
        return overlay
    
    def _process_results(self, result) -> List[Dict[str, Any]]:
        """Process YOLO results into serializable format."""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes
        masks = result.masks
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            bbox = [round(coord, 2) for coord in bbox]
            
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = self.model.names.get(class_id, f"class_{class_id}")
            confidence = float(boxes.conf[i].cpu().numpy())
            
            mask_points = 0
            if masks is not None and i < len(masks.xy):
                mask_points = len(masks.xy[i])
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "mask_points": mask_points
            })
        
        return detections


class SMPSegmenter(BaseSegmenter):
    """
    Segmentation Models Pytorch (SMP) wrapper.
    Supports UNet, DeepLabV3+, and other SMP architectures.
    """
    
    def __init__(self, model_config: Dict[str, Any], confidence: float = 0.5, device: str = None):
        super().__init__(model_config, confidence, device)
        self.model_type = model_config.get("type", "unet")
        self.encoder = model_config.get("encoder", "resnet34")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the SMP model from checkpoint."""
        try:
            import segmentation_models_pytorch as smp
            
            # Create model architecture
            if self.model_type == "unet":
                self.model = smp.Unet(
                    encoder_name=self.encoder,
                    encoder_weights=None,  # We'll load our own weights
                    in_channels=3,
                    classes=1
                )
            elif self.model_type == "deeplabv3":
                self.model = smp.DeepLabV3Plus(
                    encoder_name=self.encoder,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1
                )
            else:
                raise ValueError(f"Unknown SMP model type: {self.model_type}")
            
            # Load weights
            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
                print(f"✓ {self.model_type.upper()} model loaded: {self.model_path}")
            else:
                print(f"⚠ Checkpoint not found: {self.model_path}, using random weights")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Using device: {self.device.upper()}" + 
                  (f" ({torch.cuda.get_device_name(0)})" if self.device == 'cuda' else ""))
        except ImportError:
            raise RuntimeError("segmentation_models_pytorch not installed. Run: pip install segmentation-models-pytorch")
        except Exception as e:
            raise RuntimeError(f"Failed to load SMP model: {e}")
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for SMP model."""
        # Resize
        img = cv2.resize(frame, (self.img_size, self.img_size))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension and convert to tensor
        tensor = torch.tensor(img).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess(self, logits: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output to binary mask."""
        # Apply sigmoid
        probs = torch.sigmoid(logits)
        # Resize to original size
        probs = F.interpolate(probs, size=original_size, mode='bilinear', align_corners=False)
        # Get binary mask
        mask = (probs > self.confidence).float()
        return mask.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def segment_frame_with_visualization(self, frame: np.ndarray, 
                                          frame_index: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
        """Run segmentation and return both results and annotated frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = datetime.now()
        original_h, original_w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Inference
        logits = self.model(input_tensor)
        
        # Postprocess
        mask = self._postprocess(logits, (original_h, original_w))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get raw probability for confidence calculation
        probs = torch.sigmoid(logits)
        
        # Create detection result
        num_objects = 1 if mask.sum() > 0 else 0
        detections = []
        bbox = None
        confidence = 0.0
        if num_objects > 0:
            # Find bounding box of mask
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                confidence = float(probs.max().cpu().numpy())
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                detections.append({
                    "class_id": 0,
                    "class_name": "fire",
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                    "mask_points": int(mask.sum())
                })
        
        # Create annotated frame with mask overlay, bbox, and confidence text
        annotated_frame = self._create_overlay(frame, mask, bbox, confidence, processing_time)
        
        result_dict = {
            "frame_index": int(frame_index),
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "num_objects": int(len(detections)),
            "processing_time_ms": round(float(processing_time), 2),
            "device": str(self.device),
            "model_type": self.model_type.upper()
        }
        
        return result_dict, annotated_frame
    
    def _create_overlay(self, frame: np.ndarray, mask: np.ndarray, 
                        bbox: list = None, confidence: float = 0.0,
                        processing_time: float = 0.0) -> np.ndarray:
        """Create frame with mask overlay and bounding box (YOLO-style)."""
        overlay = frame.copy()
        
        # Green color for consistency with YOLO
        color = (0, 255, 0)  # BGR - Green
        alpha = 0.4
        
        # Create colored mask overlay
        mask_bool = mask > 0
        if mask_bool.any():
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - alpha) + 
                np.array(color) * alpha
            ).astype(np.uint8)
        
        # Draw bounding box and label if detection exists
        if bbox is not None and confidence > 0:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence (like YOLO)
            label = f"fire {confidence*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(overlay, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)
        
        return overlay


class SegFormerSegmenter(BaseSegmenter):
    """
    HuggingFace SegFormer wrapper.
    Uses transformers library for semantic segmentation.
    """
    
    def __init__(self, model_config: Dict[str, Any], confidence: float = 0.5, device: str = None):
        super().__init__(model_config, confidence, device)
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the SegFormer model."""
        try:
            from transformers import SegformerForSemanticSegmentation
            
            if os.path.exists(self.model_path):
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
                print(f"✓ SegFormer model loaded: {self.model_path}")
            else:
                # Fall back to pretrained and warn
                print(f"⚠ SegFormer checkpoint not found: {self.model_path}")
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512",
                    num_labels=1,
                    ignore_mismatched_sizes=True
                )
                print("⚠ Using base pretrained model (not fine-tuned)")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Using device: {self.device.upper()}" + 
                  (f" ({torch.cuda.get_device_name(0)})" if self.device == 'cuda' else ""))
        except ImportError:
            raise RuntimeError("transformers not installed. Run: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load SegFormer model: {e}")
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for SegFormer model."""
        # Resize
        img = cv2.resize(frame, (self.img_size, self.img_size))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension and convert to tensor
        tensor = torch.tensor(img).unsqueeze(0).to(self.device)
        return tensor
    
    @torch.no_grad()
    def segment_frame_with_visualization(self, frame: np.ndarray, 
                                          frame_index: int = 0) -> Tuple[Dict[str, Any], np.ndarray]:
        """Run segmentation and return both results and annotated frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = datetime.now()
        original_h, original_w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Inference - SegFormer uses pixel_values
        outputs = self.model(pixel_values=input_tensor)
        logits = outputs.logits
        
        # Interpolate to original size
        logits = F.interpolate(logits, size=(original_h, original_w), 
                               mode='bilinear', align_corners=False)
        
        # Get probability mask
        probs = torch.sigmoid(logits)
        mask = (probs > self.confidence).float().squeeze().cpu().numpy()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create detection result
        detections = []
        bbox = None
        confidence = 0.0
        if mask.sum() > 0:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                confidence = float(probs.max().cpu().numpy())
                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                detections.append({
                    "class_id": 0,
                    "class_name": "fire",
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                    "mask_points": int(mask.sum())
                })
        
        # Create annotated frame with mask overlay, bbox, and confidence text
        annotated_frame = self._create_overlay(frame, mask, bbox, confidence, processing_time)
        
        result_dict = {
            "frame_index": int(frame_index),
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "num_objects": int(len(detections)),
            "processing_time_ms": round(float(processing_time), 2),
            "device": str(self.device),
            "model_type": "SEGFORMER"
        }
        
        return result_dict, annotated_frame
    
    def _create_overlay(self, frame: np.ndarray, mask: np.ndarray,
                        bbox: list = None, confidence: float = 0.0,
                        processing_time: float = 0.0) -> np.ndarray:
        """Create frame with mask overlay and bounding box (YOLO-style)."""
        overlay = frame.copy()
        
        # Green color for consistency with YOLO
        color = (0, 255, 0)  # BGR - Green
        alpha = 0.4
        
        # Create colored mask overlay
        mask_bool = mask > 0
        if mask_bool.any():
            overlay[mask_bool] = (
                overlay[mask_bool] * (1 - alpha) + 
                np.array(color) * alpha
            ).astype(np.uint8)
        
        # Draw bounding box and label if detection exists
        if bbox is not None and confidence > 0:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence (like YOLO)
            label = f"fire {confidence*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(overlay, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)
        
        return overlay


def create_segmenter(model_config: Dict[str, Any], confidence: float = 0.25, 
                     iou: float = 0.45, device: str = None) -> BaseSegmenter:
    """
    Factory function to create the appropriate segmenter based on model type.
    
    Args:
        model_config: Dictionary containing model configuration (type, path, etc.)
        confidence: Confidence threshold for detections
        iou: IoU threshold (only used for YOLO)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Appropriate segmenter instance
    """
    model_type = model_config.get("type", "yolo")
    
    if model_type == "yolo":
        return YOLOSegmenter(model_config, confidence=confidence, iou=iou, device=device)
    elif model_type in ["unet", "deeplabv3"]:
        return SMPSegmenter(model_config, confidence=confidence, device=device)
    elif model_type == "segformer":
        return SegFormerSegmenter(model_config, confidence=confidence, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_available_models(models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Get list of available models (those with existing checkpoint files).
    
    Args:
        models_config: Dictionary of model configurations
    
    Returns:
        Dictionary of available models with their configs
    """
    available = {}
    for name, config in models_config.items():
        path = config.get("path", "")
        if os.path.exists(path):
            available[name] = config
            available[name]["available"] = True
        else:
            available[name] = config
            available[name]["available"] = False
    return available

