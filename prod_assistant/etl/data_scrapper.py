import csv
import time
import re
import os

''' This is a famous library for parsing HTML. Imagine HTML as a tree with many branches. BeautifulSoup helps you navigate this tree to find the exact leaves (pieces of data) you're looking for. '''
from bs4 import BeautifulSoup

'''
This is a key ingredient. Many websites have security measures to detect and block automated browsers (bots). 
undetected-chromedriver is a special version of the browser driver that adds tricks to make your script look more like a real human is browsing, increasing the chance of success.
'''
import undetected_chromedriver as uc 

'''
Selenium is the main engine here. It's a tool that allows your code to take control of a web browser. 
You're importing specific helpers from it: By to tell Selenium how to find things (e.g., by class name), Keys to simulate keyboard presses (like scrolling down), and ActionChains to perform a sequence of actions.
'''
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
This class is a blueprint for your scraper. It groups all the related functions (get_top_reviews, scrape_flipkart_products, etc.) and data into one organized unit.
'''
class FlipkartScraper:
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    The __init__ method is the constructor. When you create a FlipkartScraper, this code runs first. 
    It sets up a directory named data where your output CSV file will be saved. os.makedirs(..., exist_ok=True) is a safe way to create a directory, as it won't crash if the directory already exists.
    '''
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    This method is a specialist. Its only job is to go to a single product's page and extract the top reviews.
    '''
    def get_top_reviews(self,product_url,count=2):
        
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = uc.Chrome(
            options=options,
            use_subprocess=True
        )

        if not product_url.startswith("http"):
            return "No reviews found"

        try:

            '''
            It opens the browser and navigates to the product_url.

            -> time.sleep(4): This is crucial. It tells the script to wait for 4 seconds. You need to give the page time to fully load all its content. 
            Without this, you might try to grab data that hasn't appeared yet.

            -> Popup Handling: The try/except block is a defensive move. It tries to find and click the '✕' button on a popup. If there's no popup, it simply continues without crashing.

            -> Scrolling: The for loop simulates a user pressing the END key on their keyboard four times. 
            This is a clever trick to force the website to load more content, as many modern sites use "lazy loading" where reviews and other data only appear as you scroll down.
            '''
            driver.get(product_url)
            time.sleep(4)
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
                time.sleep(1)
            except Exception as e:
                print(f"Error occurred while closing popup: {e}")

            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            '''
            -> driver.page_source: After Selenium has done its job (loading the page, closing popups, scrolling), this gets the final, fully-loaded HTML of the page.

            -> BeautifulSoup(...): This passes the HTML to BeautifulSoup, making it ready for parsing.

            -> soup.select(...): This is where the magic happens. It uses CSS selectors to find all the <div> elements on the page that match the given class names, which the developer identified as the containers for review text.

            -> Deduplication: The code uses a set called seen to make sure it doesn't save the same review text twice. This is good practice.

            -> driver.quit(): This is extremely important. It properly closes the browser window and shuts down the driver process. If you forget this, you can end up with dozens of hidden browser processes eating up your computer's memory.
            '''
            soup = BeautifulSoup(driver.page_source, "html.parser")
            review_blocks = soup.select("div._27M-vq, div.col.EPCmJX, div._6K-7Co")
            seen = set()
            reviews = []

            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break
        except Exception:
            reviews = []

        driver.quit()
        return " || ".join(reviews) if reviews else "No reviews found"

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    This is the main orchestrator. It ties everything together to perform the full search-and-scrape operation.
    '''
    def scrape_flipkart_products(self, query, max_products=1, review_count=2):
        '''
        -> It builds a search URL (e.g., .../search?q=laptop) and opens it.
        '''
        options = uc.ChromeOptions()
        driver = uc.Chrome(options=options,use_subprocess=True)
        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(4)

        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
        except Exception as e:
            print(f"Error occurred while closing popup: {e}")

        time.sleep(2)
        products = []

        '''
        -> driver.find_elements(By.CSS_SELECTOR, "div[data-id]"): This finds the list of all product containers on the search results page.

        -> It then loops through each product (item) and uses more find_element calls with specific class names (div.KzDlHZ, div.Nx9bqj) to pinpoint the exact title, price, and rating.

        -> Regular Expressions: It uses re.search and re.findall to pluck out specific pieces of information, like the number of reviews or the product ID from a URL. This is more precise than just splitting strings.

        -> Putting it Together: For each product, it calls self.get_top_reviews() to fetch the reviews for that item. This is excellent design—reusing the specialized function we just looked at.

        -> Finally, it returns a list where each item is another list containing all the scraped details of one product.
        '''
        items = driver.find_elements(By.CSS_SELECTOR, "div[data-id]")[:max_products]
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, "div.KzDlHZ").text.strip()
                price = item.find_element(By.CSS_SELECTOR, "div.Nx9bqj").text.strip()
                rating = item.find_element(By.CSS_SELECTOR, "div.XQDdHH").text.strip()
                reviews_text = item.find_element(By.CSS_SELECTOR, "span.Wphh3N").text.strip()
                match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                total_reviews = match.group(0) if match else "N/A"

                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                product_link = href if href.startswith("http") else "https://www.flipkart.com" + href
                match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                product_id = match[0] if match else "N/A"
            except Exception as e:
                print(f"Error occurred while processing item: {e}")
                continue

            top_reviews = self.get_top_reviews(product_link, count=review_count) if "flipkart.com" in product_link else "Invalid product URL"
            products.append([product_id, title, rating, total_reviews, price, top_reviews])

        driver.quit()
        return products

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def save_to_csv(self, data, filename="product_reviews.csv"):
        """Save the scraped product reviews to a CSV file."""
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # filename includes subfolder like 'data/product_reviews.csv'
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # plain filename like 'output.csv'
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)
        