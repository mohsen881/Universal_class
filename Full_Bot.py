from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler
import requests

TOKEN="6149904211:AAGcgDoU56ClA4RMY3SeLWPAPgX0y3iNlp4"
CHAT_ID="-1001275308642"
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  level=logging.INFO)
logger = logging.getLogger(__name__)

CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
CHROMEDRIVER_PATH = ChromeDriverManager().install()
WINDOW_SIZE = "1920,1080"

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.binary_location = CHROME_PATH



def Check_Site():
    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,chrome_options=chrome_options)
    #url launch
    driver.get("https://appointment.bmeia.gv.at/")

    element = driver.find_element(By.ID,"Office")
    element.send_keys("TEHERAN")
    Bottem=driver.find_element(By.NAME,'Command')
    Bottem.click()
    #time.sleep(5)
    #################
    element = driver.find_element(By.ID,"CalendarId")
    element.send_keys('J')
    #time.sleep(5)
    #################
    Bottem=driver.find_element(By.XPATH,"//form/table[2]/tbody/tr[3]/td[2]/input[2]")
    Bottem.click()
    #time.sleep(5)
    #################
    Bottem=driver.find_element(By.XPATH,"//form/table[2]/tbody/tr[4]/td[2]/input[2]")
    Bottem.click()
    #time.sleep(5)
    #################
    Bottem=driver.find_element(By.XPATH,"//form/input[6]")
    Bottem.click()
    #time.sleep(5)
    ###############
    Note=driver.find_element(By.XPATH,"//p")
    if Note.text=="For your selection there are unfortunately no appointments available":
        print("There is no Time for jobseeker now")
        return False
    else:
        print("Lets Get The Time")
        return True
    
def TelegBot(message):
    apiToken = TOKEN
    chatID = CHAT_ID
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)
   
while True:
    try:
        if Check_Site()==True:
            TelegBot("Site is open for Booking")
        
    except ValueError:
        TelegBot("Oops! Some Error Occurred")
        print("Oops! Some Fatal Error Occurred")
