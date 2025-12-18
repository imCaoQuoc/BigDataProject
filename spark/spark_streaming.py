"""
SE363 - Fire Detection System with YOLO (Ultralytics)
Spark Streaming consumer for real-time fire segmentation/detection
"""

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, col
import pandas as pd
import json
import sys
import base64
import numpy as np
import cv2
import os
import torch
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Global variables for model loading
MODEL_PATH = "/opt/spark/model/yolo/best.pt"
IMG_SIZE = 640 # YOLO usually works best with 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None

def load_fire_model():
    global _model
    if _model is not None:
        return _model
    
    print(f"[FireModel] Loading YOLO from {MODEL_PATH} on {DEVICE}...")
    
    if os.path.exists(MODEL_PATH):
        try:
            _model = YOLO(MODEL_PATH)
            _model.to(DEVICE)
            print(f"[FireModel] ✅ Loaded YOLO weights.")
        except Exception as e:
            print(f"[FireModel] ⚠️ Error loading weights: {e}")
            # Fallback (optional, or just fail)
            _model = None
    else:
        print(f"[FireModel] ⚠️ Model file not found at {MODEL_PATH}")
        _model = None
    
    return _model

# Define Schema for UDF Output
udf_schema = T.StructType([
    T.StructField("fire_detected", T.BooleanType()),
    T.StructField("fire_percentage", T.FloatType()),
    T.StructField("confidence", T.FloatType()),
    T.StructField("image_data", T.StringType())
])

@pandas_udf(udf_schema)
def detect_fire_udf(frames: pd.Series) -> pd.DataFrame:
    model = load_fire_model()
    results_list = []
    
    if model is None:
        # Return empty results if model failed to load
        return pd.DataFrame([
            {"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None}
            for _ in range(len(frames))
        ])

    for frame_b64 in frames:
        if not frame_b64 or pd.isna(frame_b64):
            results_list.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
            continue
        
        try:
            # Decode
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results_list.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
                continue
            
            # Inference with YOLO (RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # conf=0.25 is default
            results = model.predict(img_rgb, conf=0.25, verbose=False, device=DEVICE)
            result = results[0]
            
            # Metrics
            fire_detected = len(result.boxes) > 0
            confidence = 0.0
            if fire_detected:
                confidence = float(result.boxes.conf.mean().cpu().numpy())
            
            # Calculate Percentage
            fire_pct = 0.0
            h, w = img.shape[:2]
            total_pixels = h * w
            
            # Use Box Area for Percentage
            if fire_detected:
                area_sum = 0
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    area_sum += (x2-x1) * (y2-y1)
                fire_pct = min(100.0, float((area_sum / total_pixels) * 100))

            # Visualization (Only if fire detected)
            image_data_b64 = None
            if fire_detected:
                # result.plot() returns BGR numpy array because input cv2 image was used? 
                # Ultralytics plot() usually returns BGR for cv2 compatibility.
                # Regardless, we ensure it looks right.
                res_plotted = result.plot() 
                _, buffer = cv2.imencode('.jpg', res_plotted, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_data_b64 = base64.b64encode(buffer).decode('utf-8')

            results_list.append({
                "fire_detected": fire_detected,
                "fire_percentage": fire_pct,
                "confidence": confidence,
                "image_data": image_data_b64
            })
            
        except Exception as e:
            print(f"[FireDetection] ⚠️ Error: {e}")
            results_list.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
            
    return pd.DataFrame(results_list)

def main():
    spark = (SparkSession.builder
        .appName("FireSegmentation_Streaming")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Read from Kafka
    df_stream = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "kafka:9092")
        .option("subscribe", "fire-frames")
        .option("startingOffsets", "earliest") # Consume missing data
        .option("maxOffsetsPerTrigger", 50) # Process small batches to avoid OOM
        .load())
    
    frame_schema = T.StructType([
        T.StructField("camera_id", T.StringType()),
        T.StructField("frame_number", T.IntegerType()),
        T.StructField("timestamp", T.DoubleType()),
        T.StructField("frame", T.StringType()),
        T.StructField("width", T.IntegerType()),
        T.StructField("height", T.IntegerType())
    ])
    
    df_parsed = df_stream.selectExpr("CAST(value AS STRING) as json_data") \
        .select(F.from_json("json_data", frame_schema).alias("data")) \
        .select("data.*")
    
    # Apply Inference
    df_processed = df_parsed.withColumn("detection", detect_fire_udf(col("frame")))
    
    df_final = df_processed.select(
        "camera_id",
        "frame_number",
        F.from_unixtime("timestamp").cast("timestamp").alias("detection_time"),
        "detection.fire_detected",
        "detection.fire_percentage",
        "detection.confidence",
        F.col("detection.image_data").alias("image_base64")
    )
    
    # Filter: Only keep frames with fire detected
    df_fire_only = df_final.filter(col("fire_detected") == True)
    
    # Write to Postgres
    def write_to_postgres(batch_df, batch_id):
        # Cache to prevent re-computation
        batch_df.persist() 
        count = batch_df.count()
        print(f"[Batch {batch_id}] Processing {count} FIRE frames...")
        
        if count > 0:
            try:
                (batch_df.write
                    .format("jdbc")
                    .option("url", "jdbc:postgresql://postgres:5432/airflow")
                    .option("dbtable", "fire_detections")
                    .option("user", "airflow")
                    .option("password", "airflow")
                    .option("driver", "org.postgresql.Driver")
                    .mode("append")
                    .save())
                print(f"[Batch {batch_id}] ✅ Written to DB")
            except Exception as e:
                print(f"[Batch {batch_id}] ❌ DB Error: {e}")
        
        batch_df.unpersist()

    query = (df_fire_only.writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .trigger(processingTime="1 seconds") # Faster trigger for GPU
        .start())
    
    query.awaitTermination()

if __name__ == "__main__":
    main()
