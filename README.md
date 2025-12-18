# 🎥 Video Segmentation System with Kafka Streaming

Real-time video segmentation system using multiple deep learning models (YOLO, DeepLabV3+, SegFormer, UNet) with Apache Kafka streaming and Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-brightgreen.svg)

## 📋 Features

- **Multi-Model Support**: YOLO, DeepLabV3+, SegFormer, UNet segmentation models
- **Real-time Processing**: GPU-accelerated video frame segmentation
- **Kafka Streaming**: Stream segmentation results to Apache Kafka
- **Interactive UI**: Streamlit-based web interface with real-time visualization
- **Live Monitoring**: Real-time Kafka logs, statistics, and annotated frames

## 🏗️ Architecture

```
┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Video Input   │────▶│ Segmentation    │────▶│ Kafka Topic  │
│  (Upload)      │     │ (GPU Models)    │     │ (Producer)   │
└────────────────┘     └─────────────────┘     └──────┬───────┘
                                                       │
┌────────────────┐     ┌─────────────────┐             │
│  Streamlit UI  │◀────│ Kafka Consumer  │◀────────────┘
│  (Real-time)   │     │ (Log Display)   │
└────────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
BigDataProject/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration (models, Kafka, settings)
├── docker-compose.yml      # Kafka & Zookeeper setup
├── requirements.txt        # Python dependencies
├── modules/
│   ├── segmentation.py     # Multi-model segmentation logic
│   ├── video_processor.py  # Video frame extraction
│   ├── kafka_producer.py   # Kafka message producer
│   └── kafka_consumer.py   # Kafka message consumer
├── model/                  # Model checkpoints (not included)
│   ├── yolo/best.pt
│   ├── deeplabv3/best.pth
│   ├── segformer/
│   └── unet/best.pth
└── video/                  # Sample videos (not included)
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- Docker & Docker Compose (for Kafka)

### Step 1: Clone Repository

```bash
git clone https://github.com/imCaoQuoc/BigDataProject.git
cd BigDataProject
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 12.1 (recommended)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download Model Weights

Download the trained model weights and place them in the `model/` directory:

| Model | Path | Description |
|-------|------|-------------|
| YOLO | `model/yolo/best.pt` | YOLO Segmentation |
| DeepLabV3+ | `model/deeplabv3/best.pth` | DeepLabV3+ (ResNet50) |
| SegFormer | `model/segformer/` | SegFormer (HuggingFace format) |
| UNet | `model/unet/best.pth` | UNet (ResNet34) |

### Step 6: Start Kafka

```bash
docker-compose up -d
```

This will start:
- Zookeeper on port `2181`
- Kafka broker on port `9092`

## 🎮 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### How to Use

1. **Upload Video**: Click "Upload video" in the sidebar
2. **Select Model**: Choose from YOLO, DeepLabV3+, SegFormer, or UNet
3. **Configure Settings**: Adjust frame skip and max frames
4. **Connect Kafka**: Click "Connect" to start receiving logs
5. **Start Processing**: Click "Start" to begin segmentation
6. **Monitor Results**: View real-time annotated frames and Kafka logs

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model paths
MODELS = {
    "yolo": {"path": "path/to/yolo/best.pt", ...},
    "deeplabv3": {"path": "path/to/deeplabv3/best.pth", ...},
    ...
}

# Kafka settings
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "segmentation-results"

# Processing settings
FRAME_SKIP = 0      # Process every Nth frame
MAX_FRAMES = 0      # 0 = unlimited
```

## 🐳 Docker Commands

```bash
# Start Kafka
docker-compose up -d

# Stop Kafka
docker-compose down

# View logs
docker-compose logs -f

# Reset Kafka data
docker-compose down -v
```

## 📊 Supported Models

| Model | Type | Input Size | Description |
|-------|------|------------|-------------|
| YOLO | Instance Segmentation | 640x640 | Fast, real-time detection |
| DeepLabV3+ | Semantic Segmentation | 512x512 | High accuracy, ResNet50 encoder |
| SegFormer | Semantic Segmentation | 512x512 | Transformer-based, HuggingFace |
| UNet | Semantic Segmentation | 512x512 | Classic architecture, ResNet34 encoder |

## 🛠️ Troubleshooting

### Kafka Connection Failed
- Ensure Docker is running: `docker ps`
- Check Kafka logs: `docker-compose logs kafka`
- Restart Kafka: `docker-compose restart`

### GPU Not Detected
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Model Not Found
- Verify model paths in `config.py`
- Ensure model files are downloaded to `model/` directory

## 📄 License

This project is for educational purposes.

## 👥 Authors

- [imCaoQuoc](https://github.com/imCaoQuoc)
