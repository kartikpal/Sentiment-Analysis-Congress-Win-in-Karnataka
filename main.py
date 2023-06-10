from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

data = []

path = 'C:/Users/pal28/.cache/selenium/chromedriver/win32/114.0.5735.90/chromedriver.exe'
service = Service(executable_path=path)
url = 'https://www.youtube.com/watch?v=iDLfq9Mm2G8'

with webdriver.Chrome(service=service) as driver:
    wait = WebDriverWait(driver, 10)
    driver.get(url)
    driver.maximize_window()

    scroll_count = 1000  # Number of times to scroll down
    for _ in range(scroll_count):
        wait.until(EC.visibility_of_element_located((By.TAG_NAME, 'body'))).send_keys(Keys.END)
        time.sleep(3)
    
    for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text"))):
        data.append(comment.text)

print(data)
print('Comments scraped')

import pandas as pd   
df = pd.DataFrame(data, columns=['comment'])

df.to_csv('comments.csv', index=False)
