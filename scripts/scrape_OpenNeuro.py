import requests
from bs4 import BeautifulSoup

dataset_id = "ds004902"
url = f"https://openneuro.org/datasets/{dataset_id}/"
# url = 'https://www.detik.com/search/searchall?query=KPK'
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")
articles = soup.findAll("div", {"class": "list media_rows list-berita"})
links = []

for article in articles:

    hrefs = article.find_all("a", href=True)
    for href in hrefs:
        links.append(href["href"])

print(links)

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager


# def scrape_openneuro_with_selenium(dataset_id):
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")

#     service = Service(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=chrome_options)

#     url = f"https://openneuro.org/datasets/{dataset_id}/versions/"
#     # versions/
#     driver.get(url)

#     wait = WebDriverWait(driver, 10)
#     try:
#         version_element = wait.until(
#             EC.presence_of_element_located(
#                 (By.CSS_SELECTOR, "div[class*='version-'] h1")
#             )
#         )  # [REVISED]
#         version_number = version_element.text.split()[-1]  # [REVISED]
#     except Exception:
#         version_number = "Version number not found"
#     finally:
#         driver.quit()

#     return version_number


# dataset_id = "ds004902"
# version_number = scrape_openneuro_with_selenium(dataset_id)
# print(
#     f"The latest version number for dataset {dataset_id} is: {version_number}"
# )
