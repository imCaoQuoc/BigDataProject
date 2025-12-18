# Video Segmentation System Configuration

# ============================================
# MODEL CONFIGURATION
# ============================================
# Multi-model support: Maps model names to their configurations
# Supported types: "yolo", "deeplabv3", "segformer", "unet"

MODELS = {
    "yolo": {
        "type": "yolo",
        "path": r"D:\BigDataProject\model\yolo\best.pt",
        "description": "YOLO Segmentation",
        "img_size": 640,  # YOLO input size
    },
    "deeplabv3": {
        "type": "deeplabv3",
        "path": r"D:\BigDataProject\model\deeplabv3\best.pth",
        "description": "DeepLabV3+ (ResNet50)",
        "img_size": 512,  # SMP default
        "encoder": "resnet50",
    },
    "segformer": {
        "type": "segformer",
        "path": r"D:\BigDataProject\model\segformer",  # HuggingFace format directory
        "description": "SegFormer (nvidia/segformer-b0)",
        "img_size": 512,
    },
    "unet": {
        "type": "unet",
        "path": r"D:\BigDataProject\model\unet\best.pth",
        "description": "UNet (ResNet34)",
        "img_size": 512,  # SMP default
        "encoder": "resnet34",
    }
}

# Default model to use
DEFAULT_MODEL = "yolo"

# Legacy: single model path (for backward compatibility)
MODEL_PATH = MODELS[DEFAULT_MODEL]["path"]

# Inference settings
DEFAULT_CONFIDENCE = 0.25  # Minimum confidence threshold
DEFAULT_IOU = 0.45  # IoU threshold for NMS

# ============================================
# KAFKA CONFIGURATION
# ============================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "segmentation-results"
KAFKA_GROUP_ID = "streamlit-consumer"

# ============================================
# VIDEO PROCESSING CONFIGURATION
# ============================================
FRAME_SKIP = 0  # Process every Nth frame for performance
MAX_FRAMES = 0  # Maximum number of frames to process (0 = unlimited)
