import psycopg2
import os

def check_db():
    try:
        conn = psycopg2.connect(
            dbname="airflow",
            user="airflow",
            password="airflow",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        
        # Check Airflow 'log' table
        try:
            cur.execute("SELECT count(*) FROM log;")
            print(f"Airflow 'log' table rows: {cur.fetchone()[0]}")
        except Exception as e:
            print(f"Error checking 'log' table: {e}")
            conn.rollback()

        # Check 'fire_detections' table
        try:
            cur.execute("SELECT count(*) FROM fire_detections;")
            print(f"'fire_detections' table rows: {cur.fetchone()[0]}")
        except Exception as e:
            print(f"Error checking 'fire_detections' table: {e}")
            conn.rollback()
            
        cur.close()
        conn.close()
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    check_db()
