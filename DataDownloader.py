import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from selenium.common.exceptions import TimeoutException

driver = webdriver.Chrome()
driver.maximize_window()
URL = "https://tr.investing.com/currencies/usd-try-historical-data"
data_list = []

def close_popup(second):
    try:
        close_button = WebDriverWait(driver, second).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "svg[data-test='sign-up-dialog-close-button']"))
        )
        close_button.click()
    except TimeoutException:
        pass


driver.get(URL)
close_popup(5)
scroll_amount = 10
pause_time = 0.05

thisDay = datetime.now()
thisDate = thisDay.strftime("%d.%m.%Y")
thisDate2 = (thisDay - relativedelta(months=1)).strftime("%Y-%m-%d")
print(thisDate)

for _ in range(75):
    driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
    time.sleep(pause_time)

close_popup(5)

try:
    dateBox = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '- {thisDate}')]"))
    )
    dateBox.click()
    close_popup(1)
    time.sleep(1)
    firstDateBox = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, f'(//input[@type="date" and @value="{thisDate2}"])[1]'))
    )
    firstDateBox.click()
    close_popup(1)
    for i in range(700):
        firstDateBox.send_keys(Keys.ARROW_UP)
    time.sleep(1)
    apply_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//span[text()='Uygula']")))
    close_popup(1)
    apply_button.click()
    print("BUTONA BAŞARIYLA TIKLANDI.")
except:
    print("BUTONA TIKLANAMADI.")

close_popup(5)
time.sleep(20)
goldData = driver.find_elements(By.CSS_SELECTOR,".datatable_cell__LJp3C")
for i in range(26, len(goldData), 1):
    if "%" not in goldData[i].text and "%" not in goldData[i+1].text:
        counter = (i - 26) // 3 +1
        close_popup(0)
        try:
            if counter % 200 == 0:
              print("ÇEKİLEN VERİ SAYISI :",counter)
            tarih = goldData[i].text
            fiyat= goldData[i + 1].text
            data_list.append({"Tarih": tarih, "Fiyat": fiyat})
        except (ValueError, IndexError) as e:
            print(f"Veri işlenemedi: {e}")
            continue
close_popup(5)
print("Veriler Dosyalanıyor")
df = pd.DataFrame(data_list)
df.to_csv('usd_verileri2.csv', index = False)
time.sleep(10)