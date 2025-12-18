"""
Fire Detection Dashboard
Real-time visualization of fire segmentation results reading from PostgreSQL.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
from sqlalchemy import create_engine
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="🔥 Fire Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Database connection string
# Use environment variable or default to docker service
DB_URL = os.getenv("DB_URL", "postgresql://airflow:airflow@postgres:5432/airflow")

@st.cache_resource
def get_db_engine():
    """Get SQLAlchemy engine"""
    try:
        return create_engine(DB_URL, pool_pre_ping=True)
    except Exception as e:
        st.error(f"Database engine creation failed: {e}")
        return None

def fetch_detections(limit=100):
    """Fetch recent fire detections"""
    query = f"""
        SELECT 
            camera_id,
            frame_number,
            detection_time,
            fire_detected,
            ROUND(fire_percentage::numeric, 2) as fire_percentage,
            ROUND(confidence::numeric, 3) as confidence
        FROM fire_detections
        ORDER BY detection_time DESC
        LIMIT {limit}
    """
    try:
        engine = get_db_engine()
        if engine is None:
            return pd.DataFrame()
            
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            if not df.empty:
                df['detection_time'] = pd.to_datetime(df['detection_time'])
            return df
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

def fetch_statistics():
    """Fetch aggregated statistics"""
    query = """
        SELECT 
            COUNT(*) as total_frames,
            SUM(CASE WHEN fire_detected THEN 1 ELSE 0 END) as fire_frames,
            AVG(fire_percentage) as avg_fire_percentage,
            camera_id
        FROM fire_detections
        GROUP BY camera_id
    """
    try:
        engine = get_db_engine()
        if engine is None: return pd.DataFrame()
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame()

def fetch_fire_images(limit=6):
    """Fetch recent fire detection images with masks"""
    query = f"""
        SELECT 
            camera_id,
            frame_number,
            detection_time,
            fire_percentage,
            image_base64
        FROM fire_detections
        WHERE fire_detected = true
        AND image_base64 IS NOT NULL
        ORDER BY detection_time DESC
        LIMIT {limit}
    """
    try:
        engine = get_db_engine()
        if engine is None: return pd.DataFrame()
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Image query failed: {e}")
        return pd.DataFrame()

# Main app
def main():
    st.title("🔥 Fire Detection Dashboard")
    st.markdown("Real-time fire segmentation monitoring integration (Kafka -> Spark -> Postgres -> Streamlit)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        refresh_interval = st.slider("Refresh Interval (seconds)", 2, 60, 5)
        data_limit = st.number_input("History Size", 50, 1000, 100)
        
        st.divider()
        st.info("Reads data from Postgres 'fire_detections' table populated by Spark Streaming.")
        
        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Data Fetching
    df = fetch_detections(limit=data_limit)
    stats_df = fetch_statistics()
    fire_images_df = fetch_fire_images(limit=6)

    # Metrics Row
    if not stats_df.empty:
        total_frames = stats_df['total_frames'].sum()
        total_fire = stats_df['fire_frames'].sum()
        st.metric("Total Frames Processed", f"{total_frames:,}", delta=f"{len(df)} recent")
    else:
        st.warning("Waiting for data...")
        time.sleep(refresh_interval)
        st.rerun()
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent Activity")
        st.dataframe(df, use_container_width=True, height=300)

    with col2:
        st.subheader("Camera Stats")
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Fire Gallery & Live Feed
    tab1, tab2 = st.tabs(["📺 Live Feed", "🖼️ Detection Gallery"])
    
    with tab1:
        st.subheader("🔴 Live Processed Stream")
        live_placeholder = st.empty()
        
        # Get the absolute single latest frame with image data
        latest_frame = fetch_fire_images(limit=1)
        if not latest_frame.empty:
            row = latest_frame.iloc[0]
            display_img = f"data:image/jpeg;base64,{row['image_base64']}"
            live_placeholder.image(
                display_img, 
                caption=f"Camera: {row['camera_id']} | {row['detection_time']} | Conf: {row['fire_percentage']:.1f}%",
                use_container_width=True
            )
        else:
            live_placeholder.info("Waiting for live data stream...")

    with tab2:
        st.subheader("🔥 Latest Detections History")
        if not fire_images_df.empty:
            cols = st.columns(3)
            for idx, row in fire_images_df.iterrows():
                col = cols[idx % 3]
                with col:
                    st.image(
                        f"data:image/jpeg;base64,{row['image_base64']}", 
                        caption=f"{row['camera_id']} | Conf: {row['fire_percentage']:.1f}% | {row['detection_time']}",
                        use_container_width=True
                    )
        else:
            st.info("No fire images detected yet.")

    # Auto Refresh
    time.sleep(refresh_interval)
    st.rerun()

if __name__ == "__main__":
    main()
