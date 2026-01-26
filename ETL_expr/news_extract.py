import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime

# 설정 값
RSS_URL = "https://www.mk.co.kr/rss/30000001/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_rss_article_list(rss_url):
    """RSS 피드에서 기사 목록(ID, 제목, URL) 추출"""
    try:
        response = requests.get(rss_url, headers=HEADERS)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"RSS 접속 실패: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'xml')
        items = soup.find_all('item')

        articles = []
        for item in items:
            articles.append({
                "id": item.find('no').text if item.find('no') else "N/A",
                "title": item.find('title').text if item.find('title') else "제목 없음",
                "url": item.find('link').text if item.find('link') else "",
                "pub_date": item.find('pubDate').text if item.find('pubDate') else ""
            })
        return articles
    except Exception as e:
        print(f"RSS 수집 에러: {e}")
        return []

def extract_article_content(url):
    """개별 URL에서 기사 본문 추출"""
    if not url:
        return ""
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 본문 컨테이너 (매경 기준: div.news_cnt_detail_wrap)
        content_container = soup.select_one('div.news_cnt_detail_wrap')
        
        if not content_container:
            return ""

        # 불필요한 요소 제거 (광고, 버튼, 피겨 캡션 등)
        for detail in content_container.select('.ad-slot, .btn, figcaption, figure, .thumb_area'):
            detail.decompose()

        # 텍스트 추출: strip=True와 separator='\n'을 사용해 줄바꿈 유지 및 공백 제거
        # <p> 태그가 없더라도 텍스트 노드들을 줄바꿈으로 구분해 가져옴
        full_text = content_container.get_text(separator='\n', strip=True)
        return full_text

    except Exception as e:
        print(f"본문 크롤링 에러 ({url}): {e}")
        return ""

def save_to_json(data, filename=None, output_dir=None):
    """데이터를 JSON 파일로 저장"""
    if not filename:
        # 파일명이 없으면 현재 시간을 기준으로 생성 (예: news_20260122_2345.json)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"mk_news_{timestamp}.json"

    # 저장 경로 설정 (절대 경로 사용)
    base_dir = "/opt/airflow"
    output_dir = os.path.join(base_dir, "result/airflow")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_path = os.path.abspath(os.path.join(output_dir, filename))

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"성공: {len(data)}건의 기사가 {full_path}에 저장되었습니다.")

def run_etl(output_dir=None):
    """전체 ETL 프로세스 실행"""
    print("1. RSS 기사 목록 수집 중...")
    articles = get_rss_article_list(RSS_URL)
    
    # 노트북처럼 상위 5개만 테스트하려면 articles[:5]로 수정 가능
    target_articles = articles[:5] 
    
    final_data = []
    
    print(f"2. 본문 크롤링 시작 (대상: {len(target_articles)}건)...")
    for article in target_articles:
        print(f"   - 수집 중: {article['title'][:20]}...")
        content = extract_article_content(article['url'])
        
        # 원본 데이터에 본문 내용 추가
        article['content'] = content
        article['collected_at'] = datetime.now().isoformat()
        final_data.append(article)
    
    print("3. JSON 파일 저장 중...")
    save_to_json(final_data)

if __name__ == "__main__":
    run_etl()