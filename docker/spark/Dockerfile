FROM apache/spark:3.5.1

USER root

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Spark 3.5.1 image uses Python 3.8. Pin typing-extensions to support it.
RUN pip install "typing-extensions<4.13"
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir opencv-python-headless pandas numpy psycopg2-binary kafka-python pyarrow ultralytics

USER spark
