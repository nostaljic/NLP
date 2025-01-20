from abc import ABC, abstractmethod

class NewsService(ABC):
    @abstractmethod
    def open_news_section(self, url: str):
        pass

    @abstractmethod
    def scrape_new_articles(self) -> list:
        pass

    @abstractmethod
    def get_article_content(self, url: str) -> str:
        pass

    @abstractmethod
    def leave_comment(self, url: str, comment: str):
        pass

    @abstractmethod
    def refresh_page(self):
        pass

    @abstractmethod
    def save_results(self, results: list, file_path: str):
        pass

    @abstractmethod
    def run(self, refresh_interval: int):
        pass
