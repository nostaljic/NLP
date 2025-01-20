import time
import jsonlines
from .news_service import NewsService
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class GreenNewsService(NewsService):
    def __init__(self, llm_service, summary_prompt: str, reply_prompt: str, result_file: str, log_file: str):
        self.driver = webdriver.Chrome()
        self.articles = {}
        self.llm_service = llm_service
        self.summary_prompt = summary_prompt
        self.reply_prompt = reply_prompt
        self.result_file = result_file
        self.log_file = log_file  # 로그 파일 경로
        self.results = []
    # 로그인 시간 확보를 위한 대기 함수
    def wait_for_login(self, duration: int = 15):
        print(f"로그인을 위해 {duration // 60}분 동안 대기합니다...")
        time.sleep(duration)
        print("로그인 시간이 종료되었습니다. 작업을 시작합니다.")

    def open_news_section(self, url: str):
        self.driver.get(url)
        with jsonlines.open(self.log_file, mode='a') as log_writer:
            log_writer.write({"action": "open_news_section", "url": url})

    def scrape_new_articles(self) -> list:
        new_articles = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, "._NLOG_IMPRESSION")
        for el in elements:
            href = el.get_attribute("href")
            if href not in self.articles:
                self.articles[href] = {"processed": False}
                new_articles.append(href)

        with jsonlines.open(self.log_file, mode='a') as log_writer:
            log_writer.write({"action": "scrape_new_articles", "new_articles_count": len(new_articles)})

        return new_articles

    def get_article_content(self, url: str) -> str:
        try:
            self.driver.get(url)
            content = self.driver.find_element(By.CSS_SELECTOR, "#dic_area").text
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "get_article_content", "url": url, "status": "success"})
            return content
        except Exception as e:
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "get_article_content", "url": url, "status": "error", "error": str(e)})
            return ""

    def leave_comment(self, url: str, comment: str):
        try:
            #self.driver.get(url)

            # 댓글 입력 창 대기
            wait = WebDriverWait(self.driver, 10)
            textarea = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#cbox_module__write_textarea")))
            textarea.send_keys(comment)

            # 댓글 입력 버튼 대기 및 클릭
            submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".__uis_cbox_module_writeButton")))
            submit_button.click()
            time.sleep(1)
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "leave_comment", "url": url, "status": "success", "comment": comment})
        except Exception as e:
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "leave_comment", "url": url, "status": "error", "error": str(e)})

    def refresh_page(self):
        self.driver.refresh()
        time.sleep(5)
        with jsonlines.open(self.log_file, mode='a') as log_writer:
            log_writer.write({"action": "refresh_page", "status": "success"})

    def save_results(self):
        with jsonlines.open(self.result_file, mode='a') as result_writer:
            for result in self.results:
                result_writer.write(result)
        with jsonlines.open(self.log_file, mode='a') as log_writer:
            log_writer.write({"action": "save_results", "file_path": self.result_file, "results_count": len(self.results)})
        self.results = []  # 저장 후 결과 초기화

    def run(self, refresh_interval: int):
        try:
            while True:
                # 새 기사 수집 및 처리
                new_articles = self.scrape_new_articles()
                print(f"Found {len(new_articles)} new articles.")

                for url in new_articles:
                    content = self.get_article_content(url)
                    if not content:
                        continue
                    
                    
                    print("==================================")
                    # LLM을 이용해 요약 생성
                    summary_prompt = self.summary_prompt.replace("{content}", content[:500])
                    print(f"[S-PROMPT]{summary_prompt}")
                    summary = self.llm_service.generate_response(summary_prompt)
                    print(f"[SUMMARY]{summary}")
                    print("==================================")
                    
                    # LLM을 이용해 답글 생성
                    reply_prompt = self.reply_prompt.replace("{summary}", summary[:500])
                    print(f"[R-PROMPT]{reply_prompt}")
                    reply = self.llm_service.generate_response(reply_prompt)
                    print(f"[REPLY]{reply}")

                    # 댓글 작성
                    self.leave_comment(url, reply)

                    # 결과 저장
                    self.results.append({
                        "url": url,
                        "summary": summary,
                        "comment": reply
                    })

                    # JSON 파일에 결과 저장
                    self.save_results()

                # 새로고침 및 대기
                self.refresh_page()
                print(f"Page refreshed. Waiting for {refresh_interval} seconds...")
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "run", "status": "stopped_by_user"})
            print("Service stopped by user.")
        finally:
            self.driver.quit()
            with jsonlines.open(self.log_file, mode='a') as log_writer:
                log_writer.write({"action": "run", "status": "driver_quit"})
