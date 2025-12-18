CREATE TABLE IF NOT EXISTS fire_detections (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50),
    frame_number INTEGER,
    detection_time TIMESTAMP,
    fire_detected BOOLEAN,
    fire_percentage DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    image_base64 TEXT
);

CREATE INDEX IF NOT EXISTS idx_fire_detections_time ON fire_detections(detection_time);
