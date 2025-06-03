import time
import json
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException

# --- Configuration ---
STOCK_CODE = "THYAO"
OUTPUT_JSON = "latest_disclosure.json"
KAP_URL = "https://www.kap.org.tr/en"

# --- Setup Selenium WebDriver ---
options = Options()
# Uncomment to run headless
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)
wait = WebDriverWait(driver, 30)

try:
    # Step-1) Navigate to KAP homepage
    driver.get(KAP_URL)

    # Step-2) Dismiss cookie banner if present
    try:
        cookie_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-accept-cookie, .cookie-consent-accept"))
        )
        cookie_btn.click()
    except TimeoutException:
        pass

    # Step-3) Enter stock code in search
    search_input = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,
            "input[placeholder*='Search'], input[placeholder*='Ara'], input[name='srcbar'], input[name='all-search']"
        ))
    )
    search_input.clear()
    search_input.send_keys(STOCK_CODE)

    # Step-4) Click search button
    try:
        search_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    except NoSuchElementException:
        search_btn = driver.find_element(By.CSS_SELECTOR, ".input-group-append button")
    search_btn.click()

    # Step-5) Click the "Notifications" link in the overlay
    notifications_link = wait.until(
        EC.element_to_be_clickable((By.XPATH,
            "//a[normalize-space(text())='Notifications' or normalize-space(text())='Bildirimler']"
        ))
    )
    notifications_link.click()

    # Step-6) Click the checkbox with ID "1442091"
    try:
        checkbox = wait.until(EC.presence_of_element_located((By.ID, "1442091")))
        if checkbox.is_enabled():
            checkbox.click()
            print("Checkbox clicked.")
        else:
            print("Checkbox is disabled.")
    except (TimeoutException, ElementNotInteractableException):
        print("Checkbox could not be interacted with.")

    # Step-7) Click the link to the notification (opens in new tab)
    try:
        notif_link = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'Bildirim/1442091')]"))
        )
        notif_link.click()
        print("Notification link clicked.")
    except TimeoutException:
        print("Notification link not found or not clickable.")
        driver.quit()
        exit()

    # Step-8) Switch to the new tab
    driver.switch_to.window(driver.window_handles[-1])
    print("Switched to new tab.")

    # Step-9) Wait for and extract news content
    try:
        content_div = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.disclosureScrollableAreaScrollBar"))
        )
        news_text = content_div.text.strip()

        news_json = {
            "stock_code": STOCK_CODE,
            "news_content": news_text
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(news_json, f, ensure_ascii=False, indent=2)

        print("News content extracted and saved to", OUTPUT_JSON)
    except TimeoutException:
        print("News content could not be loaded.")

finally:
    time.sleep(3)  # Optional pause
    driver.quit()
