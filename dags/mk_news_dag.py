from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

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
    'mk_news_extraction',
    default_args=default_args,
    description='매일경제 RSS 뉴스 수집 ETL',
    schedule_interval=timedelta(hours=1),  # 1시간마다 실행
    catchup=False,
    tags=['news', 'etl'],
) as dag:

    # 1. 뉴스 추출 작업
    # 컨테이너 내의 /opt/airflow/ETL_expr/news_extract.py 경로 사용
    t1 = BashOperator(
        task_id='extract_news',
        bash_command='python /opt/airflow/ETL_expr/news_extract.py',
    )

    t1
