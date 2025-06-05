import time
import json
import requests
import os
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
OUTPUT_DIR = f"{STOCK_CODE.lower()}_notifications"
KAP_URL = "https://www.kap.org.tr/en"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print("Step 1: Navigating to KAP homepage...")
    driver.get(KAP_URL)

    # Step-2) Dismiss cookie banner if present
    print("Step 2: Dismissing cookie banner...")
    try:
        cookie_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-accept-cookie, .cookie-consent-accept"))
        )
        cookie_btn.click()
        print("Cookie banner dismissed.")
    except TimeoutException:
        print("No cookie banner found.")

    # Step-3) Enter stock code in search
    print("Step 3: Entering stock code in search...")
    search_input = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,
            "input[placeholder*='Search'], input[placeholder*='Ara'], input[name='srcbar'], input[name='all-search']"
        ))
    )
    search_input.clear()
    search_input.send_keys(STOCK_CODE)
    print(f"Stock code '{STOCK_CODE}' entered.")

    # Step-4) Click search button
    print("Step 4: Clicking search button...")
    try:
        search_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    except NoSuchElementException:
        search_btn = driver.find_element(By.CSS_SELECTOR, ".input-group-append button")
    search_btn.click()
    print("Search button clicked.")

    # Step-5) Click the "Notifications" link in the overlay
    print("Step 5: Clicking Notifications link...")
    notifications_link = wait.until(
        EC.element_to_be_clickable((By.XPATH,
            "//a[normalize-space(text())='Notifications' or normalize-space(text())='Bildirimler']"
        ))
    )
    notifications_link.click()
    print("Notifications link clicked.")

    # Wait for the notifications page to load
    time.sleep(3)

    # Step-6) Find all checkboxes on the page
    print("Step 6: Finding all checkboxes...")
    try:
        # Look for all checkbox inputs that are likely notification selectors
        checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
        
        # Filter out any system checkboxes (like select all, etc.) and keep only notification checkboxes
        notification_checkboxes = []
        for checkbox in checkboxes:
            try:
                # Check if checkbox has an ID that looks like a notification ID (numbers)
                checkbox_id = checkbox.get_attribute("id")
                if checkbox_id and checkbox_id.isdigit():
                    notification_checkboxes.append(checkbox)
            except:
                continue
        
        if not notification_checkboxes:
            print("No notification checkboxes found. Trying alternative selector...")
            # Alternative approach: look for checkboxes within notification rows
            notification_checkboxes = driver.find_elements(By.CSS_SELECTOR, "tr input[type='checkbox'], .notification-row input[type='checkbox']")
        
        print(f"Found {len(notification_checkboxes)} notification checkboxes.")
        
        if len(notification_checkboxes) == 0:
            print("No checkboxes found to process.")
            driver.quit()
            exit()

    except Exception as e:
        print(f"Error finding checkboxes: {e}")
        driver.quit()
        exit()

    # Step-7) Process each checkbox systematically
    print("Step 7: Processing each checkbox systematically...")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for i, checkbox in enumerate(notification_checkboxes, 1):
        try:
            print(f"\n--- Processing checkbox {i}/{len(notification_checkboxes)} ---")
            
            # Get checkbox ID for filename
            checkbox_id = checkbox.get_attribute("id") or f"notification_{i}"
            
            # Scroll to checkbox to ensure it's visible
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", checkbox)
            time.sleep(1)
            
            # Click the checkbox
            if checkbox.is_enabled() and checkbox.is_displayed():
                try:
                    checkbox.click()
                    print(f"Checkbox {checkbox_id} clicked.")
                except ElementNotInteractableException:
                    # Try JavaScript click if regular click fails
                    driver.execute_script("arguments[0].click();", checkbox)
                    print(f"Checkbox {checkbox_id} clicked via JavaScript.")
                
                time.sleep(2)
                
                # Look for the notification link
                try:
                    # Try to find link with the checkbox ID in href
                    notif_link = driver.find_element(By.XPATH, f"//a[contains(@href, 'Bildirim/{checkbox_id}') or contains(@href, '{checkbox_id}')]")
                except NoSuchElementException:
                    # Alternative: look for any notification link that became active
                    try:
                        notif_link = driver.find_element(By.XPATH, "//a[contains(@href, 'Bildirim/')]")
                    except NoSuchElementException:
                        print(f"No notification link found for checkbox {checkbox_id}")
                        failed_downloads += 1
                        continue
                
                # Store current window handle
                original_window = driver.current_window_handle
                
                # Click the notification link
                notif_link.click()
                print(f"Notification link clicked for {checkbox_id}.")
                
                # Wait for new tab and switch to it
                time.sleep(3)
                new_window = None
                for handle in driver.window_handles:
                    if handle != original_window:
                        new_window = handle
                        break
                
                if new_window:
                    driver.switch_to.window(new_window)
                    print("Switched to new tab.")
                    
                    # Extract news content (TEXT from the page, not downloading files)
                    try:
                        # Wait for the notification content to load
                        content_div = wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.disclosureScrollableAreaScrollBar"))
                        )
                        
                        # Extract the text content from the notification page
                        news_text = content_div.text.strip()
                        
                        if news_text:
                            # Create JSON data with the extracted text content
                            news_json = {
                                "stock_code": STOCK_CODE,
                                "checkbox_id": checkbox_id,
                                "notification_number": i,
                                "news_content": news_text,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "url": driver.current_url
                            }
                            
                            # Save to separate JSON file
                            filename = f"{OUTPUT_DIR}/notification_{checkbox_id}.json"
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(news_json, f, ensure_ascii=False, indent=2)
                            
                            print(f"News text content extracted and saved to {filename}")
                            print(f"Content preview: {news_text[:100]}...")
                            successful_downloads += 1
                        else:
                            print(f"No text content found for {checkbox_id}")
                            failed_downloads += 1
                        
                    except TimeoutException:
                        print(f"News content div could not be loaded for {checkbox_id}")
                        # Try alternative selectors for content
                        try:
                            alternative_content = driver.find_element(By.CSS_SELECTOR, ".disclosure-content, .notification-content, .content")
                            news_text = alternative_content.text.strip()
                            if news_text:
                                news_json = {
                                    "stock_code": STOCK_CODE,
                                    "checkbox_id": checkbox_id,
                                    "notification_number": i,
                                    "news_content": news_text,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "url": driver.current_url
                                }
                                filename = f"{OUTPUT_DIR}/notification_{checkbox_id}.json"
                                with open(filename, "w", encoding="utf-8") as f:
                                    json.dump(news_json, f, ensure_ascii=False, indent=2)
                                print(f"News content found with alternative selector and saved to {filename}")
                                successful_downloads += 1
                            else:
                                failed_downloads += 1
                        except:
                            print(f"Could not extract content for {checkbox_id} with any method")
                            failed_downloads += 1
                    
                    # Close the new tab and switch back to original
                    driver.close()
                    driver.switch_to.window(original_window)
                    print("Closed new tab and returned to original page.")
                    
                else:
                    print(f"New tab not opened for {checkbox_id}")
                    failed_downloads += 1
                
            else:
                print(f"Checkbox {checkbox_id} is not enabled or visible. Skipping.")
                failed_downloads += 1
                
            # Small delay before processing next checkbox
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing checkbox {i}: {e}")
            failed_downloads += 1
            
            # Ensure we're back on the original window
            try:
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue

    # Summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total checkboxes processed: {len(notification_checkboxes)}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Files saved in directory: {OUTPUT_DIR}")

finally:
    time.sleep(3)  # Optional pause
    driver.quit()
    print("Browser closed.")