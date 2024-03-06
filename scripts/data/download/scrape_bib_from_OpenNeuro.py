#!/env/bin/env python
# pip install webdriver_manager
"""
cd /tmp
wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
unzip chromedriver_linu64.zip
sudo mv chromedriver /usr/bin
pip install -U webdriver_manager
$ chromedriver --version
ChromeDriver 114.0.5735.90 (386bc09e8f4f2e025eddae123f36f6263096ae49-refs/branch-heads/5735@{#1052})
$ pip show selenium
Name: selenium
Version: 4.17.2
Summary: 
Home-page: https://www.selenium.dev
Author: 
Author-email: 
License: Apache 2.0
Location: /home/ywatanabe/proj/EEGPT/env/lib/python3.9/site-packages
Requires: certifi, trio, trio-websocket, typing_extensions, urllib3
Required-by: 
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Setup driver using the new Service object
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait

# # Setup driver (make sure to use the correct ChromeDriver executable path)
# # driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver')
# driver = webdriver.Chrome()

# Navigate to the page
# driver.get("https://openneuro.org/datasets/ds004100/versions/1.1.3")
driver.get("https://openneuro.org/datasets/ds004100/")

WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.ID, "element_id"))
)
# Wait for the 'BibTeX' button to be clickable and click it
bibtex_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='BibTeX']"))
)
bibtex_button.click()

# Wait for the BibTeX content to be visible
bibtex_content = WebDriverWait(driver, 10).until(
    EC.visibility_of_element_located(
        (By.CSS_SELECTOR, "h5.cite-content-block pre")
    )
)

# Extract the BibTeX text
bibtex_text = bibtex_content.text

# Now you can use bibtex_text as needed
print(bibtex_text)

# Don't forget to close the driver after your actions
driver.quit()

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# Function to set up the Selenium WebDriver
def setup_driver():
    # Options for the WebDriver (e.g., headless mode)
    options = webdriver.ChromeOptions()
    options.add_argument(
        "--headless"
    )  # Uncomment if you don't need a browser UI
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Set up the Chrome Service
    service = Service(ChromeDriverManager().install())

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    return driver


# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.ui import WebDriverWait


# # Function to set up the Selenium WebDriver
# def setup_driver():
#     # Options for the WebDriver (e.g., headless mode)
#     options = webdriver.ChromeOptions()
#     options.add_argument("--headless")  # Uncomment if you don't need a browser UI
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")

#     # Path to your WebDriver executable
#     driver_path = "/usr/bin/chromedriver"

#     # Initialize the WebDriver
#     driver = webdriver.Chrome(executable_path=driver_path, options=options) # TypeError: __init__() got an unexpected keyword argument 'executable_path'
#     return driver


# Function to get the BibTeX data using Selenium
def get_bibtex_with_selenium(driver, dataset_id):
    base_url = f"https://openneuro.org/datasets/{dataset_id}"
    driver.get(base_url)

    # Wait for the BibTeX button to be clickable, and then click it
    # You will need to inspect the webpage to find the correct ID or class to wait for
    bibtex_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.ID, "bibtex-button-id")
        )  # Update with the correct ID or class
    )
    bibtex_button.click()

    # Now wait for the BibTeX content to appear and then extract it
    # You will need to inspect the webpage to find the correct ID or class to wait for
    bibtex_content = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.CLASS_NAME, "bibtex-content-class")
        )  # Update with the correct ID or class
    )

    return bibtex_content.text


# Main execution
if __name__ == "__main__":
    driver = setup_driver()
    dataset_ids = ["ds004100"]  # Add all your dataset IDs here
    bibtex_entries = {}

    for dataset_id in dataset_ids:
        bibtex_entries[dataset_id] = get_bibtex_with_selenium(
            driver, dataset_id
        )

    driver.quit()

    # Print or save the BibTeX entries as needed
    for dataset_id, bibtex in bibtex_entries.items():
        print(f"BibTeX for {dataset_id}:")
        print(bibtex)
        print("\n")
