from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mk_news_full_pipeline',
    default_args=default_args,
    description='매일경제 뉴스 수집 및 모듈형 지능형 분석 ETL',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['news', 'ai', 'etl'],
) as dag:

    # 1. 뉴스 추출 (Extract)
    t1 = BashOperator(
        task_id='extract_news',
        bash_command='export PYTHONPATH=$PYTHONPATH:/opt/airflow && python /opt/airflow/app/etl/extractor.py',
    )

    # 2. 뉴스 변환 및 하이브리드 적재 (Transform & Load)
    t2 = BashOperator(
        task_id='transform_load_news',
        bash_command='export PYTHONPATH=$PYTHONPATH:/opt/airflow && python /opt/airflow/app/etl/enricher.py',
        env={
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        }
    )

    t1 >> t2
