from langchain_core.tools import BaseTool
from selenium import webdriver
from typing_extensions import override
from markdownify import markdownify as md
from readability import Document
import pymupdf
import requests
from bs4 import BeautifulSoup

def save_webpage_as_pdf(title, content_md):
    text = f"{title}{content_md}"
    text_area = pymupdf.Rect(50, 50, 550, 800)
    doc = pymupdf.open()

    while True:
        print("Creating PDF...")
        page = doc.new_page()
        remaining_text = page.insert_textbox(
            text_area,
            f"{title}{content_md}",
            fontsize=12
        )
        if remaining_text:
            break

    doc.save(f"db/{title}.pdf")
    doc.close()
    print("PDF saved successfully.")
    return f"db/{title}.pdf"


class WebScrape(BaseTool):
    """A tool that scrapes a website given the link"""
    name: str = "WebScrape"
    description: str = "A tool that can scrape a website given a link"
    @override
    def _run(self, link: str) -> str:
        allowable_domain = ["https://example.com/", "https://www.productreview.com.au/", "https://www.scorptec.com.au/",
                            "https://www.harveynorman.com.au/", "https://en.wikipedia.org/wiki/", "https://au.trustpilot.com/", "https://www.techradar.com/"]
        if any(link.startswith(domain) for domain in allowable_domain):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            try:
                response = requests.get(link, headers=headers)
                if response.status_code != 200:
                    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
                    return f"Failed to retrieve the webpage. Status code: {response.status_code}"

                soup = BeautifulSoup(response.content, 'lxml')

                page_text = soup.get_text(separator='\n', strip=True)
                print(page_text)
                if page_text:
                    #path =  save_webpage_as_pdf(soup.title.string, md(page_text))
                    print(f"#{soup.title.string}\n\n{md(page_text)}")
                    return f"#{soup.title.string}\n\n{md(page_text)}"
                else:
                    driver = webdriver.Chrome()
                    print("starting to scrape")
                    try:
                        print(f"Trying to get content {link}")
                        driver.get(link)
                        """
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        """
                        html_content = driver.page_source
                        doc  = Document(html_content)
                        title = doc.title()
                        body = doc.summary()
                        content_md = md(body)

                        save_webpage_as_pdf(title, content_md)
                        print(f"#{soup.title.string}\n\n{md(page_text)}")
                        return f"#{title}\n\n{content_md} and the webpage has been saved at /db as {title}.pdf"
                    except Exception as e:
                        return f'An error occurred during scraping: {e}'
                    finally:
                        driver.quit()
            finally:
                print("Finished scraping process.")
        else:
            return f"{link} is not in the allowable domains to visit"










