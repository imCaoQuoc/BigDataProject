from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import docker
import time
import os

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'retries': 0,
}

dag = DAG(
    'fire_detection_lifecycle',
    default_args=default_args,
    description='Orchestrate Fire Detection Pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['fire', 'streaming'],
)

def get_docker_client():
    return docker.from_env()

def start_spark_streaming_fn():
    client = get_docker_client()
    # Find spark master container
    containers = client.containers.list(filters={"name": "spark-master"})
    if not containers:
        raise Exception("Spark Master container not running")
    master = containers[0]
    
    # Exec spark-submit
    # Path is /opt/spark/bin/spark-submit in official image
    cmd = (
        "/bin/bash -c '/opt/spark/bin/spark-submit "
        "--master spark://spark-master:7077 "
        "--conf spark.driver.host=spark-master "
        "--conf spark.executor.memory=2g "
        "--conf spark.cores.max=4 "
        "--jars /opt/spark/jars_custom/spark-sql-kafka-0-10_2.12-3.5.1.jar,"
        "/opt/spark/jars_custom/kafka-clients-3.4.1.jar,"
        "/opt/spark/jars_custom/commons-pool2-2.11.1.jar,"
        "/opt/spark/jars_custom/spark-token-provider-kafka-0-10_2.12-3.5.1.jar,"
        "/opt/spark/jars_custom/postgresql-42.6.0.jar "
        "/opt/spark/app/spark_streaming.py "
        "> /opt/spark/app/spark_streaming.log 2>&1'"
    )
    
    print(f"Executing: {cmd}")
    exec_id = master.exec_run(cmd, detach=True)
    print(f"Spark Job Submitted. Exec ID: {exec_id}")
    return "Started Spark Job"

def start_ingestion_fn():
    client = get_docker_client()
    workers = client.containers.list(filters={"name": "spark-worker"})
    if not workers:
        raise Exception("Spark Worker not running")
    worker = workers[0]
    
    # Path is /opt/spark/app/ingestion/producer.py in official image mapping
    cmd = "/bin/bash -c 'python3 -u /opt/spark/app/ingestion/producer.py > /opt/spark/app/producer.log 2>&1'"
    exec_id = worker.exec_run(cmd, detach=True)
    print(f"Ingestion Submitted to worker. Exec ID: {exec_id}")
    return "Started Ingestion"

def stop_pipeline_fn():
    client = get_docker_client()
    
    # Stop Ingestion in Worker
    workers = client.containers.list(filters={"name": "spark-worker"})
    if workers:
        worker = workers[0]
        worker.exec_run("pkill -f producer.py")
        print("Killed producer on worker")
        
    # Stop Spark Streaming in Master
    masters = client.containers.list(filters={"name": "spark-master"})
    if masters:
        master = masters[0]
        master.exec_run("pkill -f spark_streaming.py")
        print("Killed spark streaming on master")

start_spark = PythonOperator(
    task_id='start_spark_streaming',
    python_callable=start_spark_streaming_fn,
    dag=dag,
)

start_ingestion = PythonOperator(
    task_id='start_ingestion',
    python_callable=start_ingestion_fn,
    dag=dag,
)

monitor_pipeline = BashOperator(
    task_id='monitor_pipeline',
    bash_command='echo "Pipeline running... Check Streamlit at port 8501"; sleep 30',
    dag=dag,
)

stop_pipeline = PythonOperator(
    task_id='stop_pipeline',
    python_callable=stop_pipeline_fn,
    dag=dag,
    trigger_rule='all_done' 
)

start_spark >> start_ingestion >> monitor_pipeline >> stop_pipeline
