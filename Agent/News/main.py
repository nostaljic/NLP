'''
Project "Insight"
프로젝트 혜안 (慧眼)
프로젝트 혜안은, 
대한민국 경제 뉴스를 참조 및 현재 대한민국에 끼칠 영향을 분석한다.
분석 결과를 포스팅하여 사람들에게 빠르고 가볍게 정보를 전달한다.
예시로 댓글 형태로 작성하여 올리는 것을 데모로 작성하였다.
'''
from service import GGUFLLMService 
from service import GreenNewsService 

# 경로 설정
#CHROMEDRIVER_PATH = "./chromedriver"
MODEL_PATH = "./ai"
MODEL_FILENAME = "qwen_q3.gguf"
PROMPT_FILE = "./prompt/prompt.json"
NEWS_URL = ""#(Example) "https://news.naver.com/section/101"
RESULT_FILE = "./data/result.json"
LOG_FILE = "./data/log.json"

def main():
    # LLM 서비스 초기화
    llm_service = GGUFLLMService(MODEL_PATH, MODEL_FILENAME)
    summary_prompt = llm_service.load_prompt(PROMPT_FILE,"summary_prompt")
    reply_prompt = llm_service.load_prompt(PROMPT_FILE,"reply_prompt")

    # GreenNewsService 초기화
    news_service = GreenNewsService(
        llm_service=llm_service,
        summary_prompt=summary_prompt,
        reply_prompt=reply_prompt,
        result_file=RESULT_FILE,
        log_file=LOG_FILE
    )
    news_service.wait_for_login()
    news_service.open_news_section(NEWS_URL)
    
    # 서비스 실행
    news_service.run(refresh_interval=3600)

if __name__ == "__main__":
    main()
