'''
Refactored news processing logic, removing Tkinter UI and applying improvements.
'''
import feedparser
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException as SeleniumTimeoutException
from bs4 import BeautifulSoup
from transformers import pipeline, PipelineException
import time
import logging
import webbrowser # Keep for potential future use, but not directly in Streamlit
from telegram import Bot
from telegram.error import TimedOut, TelegramError
from telegram.request import HTTPXRequest
from dotenv import load_dotenv
import os
import json
import asyncio
import threading # Keep for potential background tasks if needed
import requests
from datetime import datetime
from requests.exceptions import Timeout, RequestException
# from retrying import retry # Removing retrying library as we implement custom logic
import torch
from newspaper import Article, ArticleException
import concurrent.futures

# Cấu hình logging (đảm bảo ghi vào file và xử lý encoding)
log_file = 'news_summary.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO, # Set to INFO, DEBUG can be too verbose for production
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    errors='replace'
)

# Tải biến môi trường từ .env
load_dotenv()

# Cấu hình Telegram và Hugging Face
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

# Danh mục tin tức
CATEGORIES = ["Chính trị", "Kinh tế", "Công nghệ", "Thể thao", "Sức khỏe", "Khác"]

# Tùy chọn số bài báo
LIMITS = ["5", "10", "15", "20", "Tất cả"]

# Tùy chọn khu vực Google News
REGIONS = {
    "Việt Nam": {"hl": "vi", "gl": "VN"},
    "Hoa Kỳ": {"hl": "en", "gl": "US"},
    "Anh": {"hl": "en", "gl": "GB"},
    "Nhật Bản": {"hl": "ja", "gl": "JP"},
    "Pháp": {"hl": "fr", "gl": "FR"}
}

# File lưu cấu hình (Streamlit might handle config differently)
# CONFIG_FILE = "user_config.json" # Keep for reference, but might not be used by Streamlit app

class NewsProcessor:
    def __init__(self):
        self.driver = None # Initialize driver as None
        self._initialize_resources()

    def _initialize_resources(self):
        '''Initializes models, Selenium driver, and Telegram bot.'''
        # Kiểm tra GPU
        self.device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'} (CUDA available: {torch.cuda.is_available()})")

        # Khởi tạo mô hình tóm tắt
        try:
            start_time = time.time()
            # Using a potentially smaller/faster model for testing if needed
            # self.summarizer = pipeline("summarization", model="google/pegasus-xsum", device=self.device, token=HF_TOKEN)
            self.summarizer = pipeline(
                "summarization",
                model="vinai/bartpho-syllable",
                clean_up_tokenization_spaces=True,
                device=self.device,
                token=HF_TOKEN
            )
            logging.info(f"Tải mô hình tóm tắt thành công trên {'GPU' if self.device == 0 else 'CPU'}. Thời gian: {time.time() - start_time:.2f}s")
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình tóm tắt: {e}", exc_info=True)
            # Propagate error to inform Streamlit UI
            raise ImportError(f"Không thể tải mô hình tóm tắt: {e}")

        # Khởi tạo mô hình Zero-Shot
        try:
            start_time = time.time()
            self.classifier = pipeline(
                "zero-shot-classification",
                # Using a recommended multi-lingual model
                model="MoritzLaurer/mDeBERTa-v3-base-xnli", 
                clean_up_tokenization_spaces=True,
                device=self.device,
                token=HF_TOKEN
            )
            logging.info(f"Tải mô hình Zero-Shot thành công trên {'GPU' if self.device == 0 else 'CPU'}. Thời gian: {time.time() - start_time:.2f}s")
        except Exception as e:
             logging.error(f"Lỗi khi tải mô hình Zero-Shot: {e}", exc_info=True)
             # Propagate error
             raise ImportError(f"Không thể tải mô hình Zero-Shot: {e}")

        # Khởi tạo Selenium (only if needed, consider lazy init)
        # For now, initialize upfront but handle errors
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox") # Important for running in containers/sandboxes
            chrome_options.add_argument("--disable-dev-shm-usage") # Important for running in containers/sandboxes
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            # Use webdriver-manager to handle driver installation/update
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            logging.info("Khởi tạo Selenium driver thành công")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo Selenium: {e}", exc_info=True)
            self.driver = None # Ensure driver is None if init fails
            # Don't raise immediately, allow fallback to newspaper3k if possible
            # raise RuntimeError(f"Không thể khởi tạo Selenium driver: {e}")
            logging.warning("Selenium driver không khởi tạo được, sẽ chỉ sử dụng newspaper3k.")


        # --- Khởi tạo Telegram Bot với cấu hình Proxy --- 
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logging.warning("Thiếu TELEGRAM_TOKEN hoặc TELEGRAM_CHAT_ID trong .env. Telegram sẽ không hoạt động.")
            self.bot = None
            # raise ValueError("Vui lòng cấu hình TELEGRAM_TOKEN và TELEGRAM_CHAT_ID trong file .env")
        else:
            proxy_enabled = os.getenv('PROXY_ENABLED', 'False').lower() == 'true'
            proxy_url = os.getenv('PROXY_URL')

            bot_request = None
            if proxy_enabled and proxy_url:
                try:
                    logging.info(f'Sử dụng proxy Telegram: {proxy_url}')
                    bot_request = HTTPXRequest(proxy_url=proxy_url, connect_timeout=10.0, read_timeout=20.0)
                    # Simple check if proxy URL seems valid (basic)
                    if not proxy_url.startswith(("socks5://", "http://", "https://")):
                         logging.warning(f"Định dạng PROXY_URL không hợp lệ: {proxy_url}. Thử không dùng proxy.")
                         bot_request = None
                except Exception as e:
                    logging.error(f'Lỗi cấu hình proxy Telegram: {e}. Thử không dùng proxy.', exc_info=True)
                    bot_request = None # Fallback không dùng proxy nếu cấu hình lỗi
            else:
                logging.info('Không sử dụng proxy Telegram.')

            try:
                # Khởi tạo Bot với request đã cấu hình (hoặc None nếu không có proxy)
                self.bot = Bot(token=TELEGRAM_TOKEN, request=bot_request)
                # Test connection by getting bot info (optional, can slow down init)
                # asyncio.run(self.bot.get_me()) # Requires running in async context or managing loop
                logging.info(f'Khởi tạo Telegram Bot thành công.')
            except TelegramError as e:
                logging.error(f'Lỗi khởi tạo Telegram Bot (có thể do token sai hoặc proxy chặn): {e}', exc_info=True)
                self.bot = None # Set bot to None if init fails
                # raise ConnectionError(f"Không thể khởi tạo Bot Telegram: {e}") from e
            except Exception as e:
                logging.error(f'Lỗi không xác định khi khởi tạo Bot Telegram: {e}', exc_info=True)
                self.bot = None
                # raise ConnectionError(f"Lỗi không xác định khi khởi tạo Bot: {e}") from e
        # --- Kết thúc phần khởi tạo Bot --- 

    def classify_article(self, title, description):
        """Phân loại bài báo bằng Zero-Shot, sử dụng nhiều ngữ cảnh hơn."""
        if not self.classifier:
             logging.warning("Classifier model not loaded. Skipping classification.")
             return "Khác"
        try:
            start_time = time.time()
            # Kết hợp tiêu đề và mô tả dài hơn (ví dụ: 250 ký tự)
            text_input = (title or "") + ". " + (description[:250] if description else "") 
            if not text_input.strip(): # Xử lý trường hợp cả title và description đều rỗng
                 logging.warning("Input rỗng cho phân loại.")
                 return "Khác"
            
            # Ensure text is not excessively long for the classifier model
            # Most models handle truncation, but explicit check can be useful
            # max_classifier_len = 512 # Example limit
            # if len(text_input) > max_classifier_len:
            #     text_input = text_input[:max_classifier_len]

            result = self.classifier(text_input, candidate_labels=CATEGORIES, multi_label=False)
            classified_label = result['labels'][0]
            logging.debug(f"Phân loại bài: {title[:30]}... -> {classified_label} (Score: {result['scores'][0]:.2f}). Thời gian: {time.time() - start_time:.2f}s")
            return classified_label
        except PipelineException as e:
            logging.error(f"Lỗi Pipeline khi phân loại bài viết: {title[:30]}...: {e}", exc_info=True)
            return "Khác"
        except Exception as e:
            logging.error(f"Lỗi không xác định khi phân loại bài viết: {title[:30]}...: {e}", exc_info=True)
            return "Khác"

    async def send_to_telegram(self, summary_text):
        """Gửi tóm tắt qua Telegram với retry và kiểm tra bot."""
        if not self.bot:
            logging.warning("Telegram Bot chưa được khởi tạo. Bỏ qua gửi tin nhắn.")
            return "Telegram Bot not initialized"
        
        error_message = None
        try:
            start_time = time.time()
            # Check network to Telegram API (optional, adds delay)
            # try:
            #     requests.get("https://api.telegram.org", timeout=5)
            #     logging.debug("Kết nối Telegram API ổn định.")
            # except RequestException as e:
            #     logging.warning(f"Mạng yếu khi kết nối Telegram: {e}")
            #     return f"Network issue connecting to Telegram: {e}"
            
            retries = 3 # Reduced retries
            for attempt in range(retries):
                try:
                    await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=summary_text, timeout=20) # Increased timeout
                    logging.info(f"Đã gửi tóm tắt qua Telegram. Thời gian: {time.time() - start_time:.2f}s")
                    return None # Success
                except TimedOut as e:
                    logging.warning(f"Thử gửi Telegram lần {attempt + 1}/{retries} thất bại (TimedOut): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(3) # Slightly longer sleep
                    else:
                        error_message = f"Telegram timed out after {retries} attempts: {e}"
                except TelegramError as e:
                    logging.error(f"Lỗi Telegram API khi gửi lần {attempt + 1}/{retries}: {e}")
                    error_message = f"Telegram API error: {e}"
                    break # Don't retry on API errors like bad request or unauthorized
                except Exception as e:
                    logging.error(f"Lỗi không xác định khi gửi Telegram lần {attempt + 1}/{retries}: {e}", exc_info=True)
                    error_message = f"Unknown error sending to Telegram: {e}"
                    # Decide whether to retry on unknown errors
                    if attempt < retries - 1:
                        await asyncio.sleep(3)
                    else:
                         error_message = f"Unknown error sending to Telegram after {retries} attempts: {e}"
            
            if error_message:
                 logging.error(f"Gửi Telegram thất bại hoàn toàn: {error_message}")
            return error_message # Return error string or None for success

        except Exception as e:
            logging.error(f"Lỗi nghiêm trọng trong hàm send_to_telegram: {e}", exc_info=True)
            return f"Critical error in send_to_telegram: {e}"

    def fetch_single_article(self, link, title_for_log=""):
        """Tải nội dung một bài báo, ưu tiên newspaper3k, fallback Selenium."""
        content = ""
        method_used = ""
        min_content_length = 150 # Increased minimum length

        # --- Thử newspaper3k trước ---
        try:
            logging.debug(f"[Article Fetch] Thử newspaper3k cho: {link}")
            article = Article(link, language='vi') # Specify language if known
            article.download()
            article.parse()
            content = article.text
            if content and len(content) >= min_content_length:
                logging.info(f"[Article Fetch] newspaper3k thành công cho: {title_for_log[:30]}...")
                method_used = "newspaper3k"
            else:
                logging.warning(f"[Article Fetch] newspaper3k trả nội dung ngắn (<{min_content_length}) cho: {title_for_log[:30]}... Length: {len(content)}. Thử Selenium.")
                content = "" # Reset content để thử Selenium
        except ArticleException as e:
            logging.warning(f"[Article Fetch] newspaper3k lỗi ({type(e).__name__}) cho: {title_for_log[:30]}... Thử Selenium. Lỗi: {e}")
            content = "" # Reset content
        except Exception as e:
            logging.error(f"[Article Fetch] Lỗi không xác định từ newspaper3k cho {link}: {e}", exc_info=True)
            content = "" # Reset content

        # --- Nếu newspaper3k thất bại và Selenium có sẵn, thử Selenium --- 
        if not content and self.driver:
            try:
                logging.debug(f"[Article Fetch] Thử Selenium cho: {link}")
                # Lấy URL thật (quan trọng vì Selenium cần URL cuối cùng)
                try:
                    response = requests.get(link, timeout=15, allow_redirects=True, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    })
                    response.raise_for_status()
                    final_url = response.url
                    logging.debug(f"[Article Fetch] URL cuối cùng cho Selenium: {final_url}")
                except RequestException as req_err:
                    logging.error(f"[Article Fetch] Không thể lấy URL cuối cùng cho Selenium ({link}): {req_err}")
                    # Don't raise, just log and return empty content
                    return "", "failed_get_final_url"

                self.driver.get(final_url)
                # Wait for body, consider more specific waits if needed
                WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                # time.sleep(3) # Avoid fixed sleeps if possible
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # Improved extraction logic (example: find main content area if possible)
                # This is highly site-dependent. Sticking to paragraphs for now.
                paragraphs = soup.find_all("p")
                extracted_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 30]) # Slightly longer min paragraph length
                
                if extracted_text and len(extracted_text) >= min_content_length:
                    content = extracted_text
                    logging.info(f"[Article Fetch] Selenium thành công cho: {title_for_log[:30]}...")
                    method_used = "selenium"
                else:
                    logging.warning(f"[Article Fetch] Selenium trả nội dung ngắn (<{min_content_length}) cho: {title_for_log[:30]}... Length: {len(extracted_text)}")
                    content = "" # Mark as failed

            except SeleniumTimeoutException as e:
                 logging.error(f"[Article Fetch] Lỗi Selenium TimeoutException cho {link}: {e}")
                 content = ""
            except WebDriverException as e:
                logging.error(f"[Article Fetch] Lỗi Selenium WebDriver ({type(e).__name__}) cho {link}: {e}", exc_info=False) # Avoid overly verbose logs for common WebDriver errors
                content = "" # Mark as failed
            except Exception as e:
                logging.error(f"[Article Fetch] Lỗi không xác định từ Selenium cho {link}: {e}", exc_info=True)
                content = "" # Mark as failed
        elif not content and not self.driver:
             logging.warning(f"[Article Fetch] newspaper3k thất bại và Selenium không có sẵn cho: {title_for_log[:30]}...")
             method_used = "newspaper3k_failed_no_selenium"

        # No manual truncation here - let the summarizer handle it.
        return content, method_used

    # Note: This function is now synchronous for easier integration with Streamlit initially.
    # Async/threading can be added back if performance becomes an issue and is managed carefully.
    def process_news(self, region, selected_category, article_limit_str, progress_callback=None):
        """Fetches, classifies, and summarizes news. Returns results and status."""
        logging.info(f"Bắt đầu process_news: region={region}, category={selected_category}, limit={article_limit_str}")
        news_by_category = {cat: [] for cat in CATEGORIES}
        processed_count = 0
        skipped_count = 0
        skipped_reasons = []
        status_message = "Bắt đầu..." 
        telegram_report = [] # Store telegram send results

        # --- 1. Fetch RSS Feed --- 
        try:
            logging.debug("Kiểm tra kết nối mạng...")
            requests.get("https://news.google.com", timeout=5)
            logging.info("Kết nối mạng ổn định.")
        except RequestException as e:
            logging.error(f"Lỗi kết nối mạng: {e}")
            return None, f"Lỗi: Không thể kết nối đến Google News ({e})", []

        region_params = REGIONS.get(region, REGIONS["Việt Nam"])
        rss_url = f"https://news.google.com/rss?hl={region_params['hl']}&gl={region_params['gl']}&ceid={region_params['gl']}:{region_params['hl']}"
        logging.info(f"Bắt đầu quét RSS: {rss_url}")
        start_time_rss = time.time()
        feed = feedparser.parse(rss_url)
        logging.info(f"Hoàn tất quét RSS. Thời gian: {time.time() - start_time_rss:.2f}s. Số mục: {len(feed.entries)}")

        if not feed.entries:
            logging.error("Không có tin tức từ Google News.")
            return None, "Không quét được tin tức từ Google News.", []

        limit = len(feed.entries) if article_limit_str == "Tất cả" else min(int(article_limit_str), len(feed.entries))
        total_entries_to_process = limit
        if progress_callback: progress_callback(0, total_entries_to_process * 2, "Đã lấy RSS feed") # Total steps = fetch + process

        articles_to_fetch = []
        for i, entry in enumerate(feed.entries[:limit]):
            articles_to_fetch.append({
                "index": i + 1,
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "description": entry.get("description", ""),
                "published": entry.get("published", datetime.now().isoformat()) # Add default published time
            })

        # --- 2. Fetch Articles in Parallel --- 
        fetched_articles_data = []
        max_workers = 5 # Adjust based on resources
        logging.info(f"Bắt đầu tải {len(articles_to_fetch)} bài báo với max_workers={max_workers}...")
        
        futures = []
        # Using ThreadPoolExecutor for I/O bound tasks (network requests, selenium waits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for article_info in articles_to_fetch:
                # logging.info(f"Đưa vào hàng đợi tải: Bài {article_info['index']} - {article_info['title'][:30]}...")
                future = executor.submit(self.fetch_single_article, article_info['link'], article_info['title'])
                futures.append((future, article_info))

            processed_futures = 0
            for future, article_info in futures:
                article_index = article_info['index']
                article_title = article_info['title']
                article_link = article_info['link']
                try:
                    content, method_used = future.result() # Wait for result
                    if content:
                        logging.info(f"Tải thành công bài {article_index} ({method_used}): {article_title[:30]}...")
                        fetched_articles_data.append({
                            "content": content,
                            "original_info": article_info
                        })
                    else:
                        logging.warning(f"Bỏ qua bài {article_index}: Không lấy được nội dung ({method_used}) - {article_title[:30]}...")
                        skipped_count += 1
                        skipped_reasons.append(f"Bài {article_index}: Không lấy được nội dung ({method_used})")
                except Exception as exc:
                    logging.error(f"Lỗi khi tải bài {article_index} ({article_link}): {exc}", exc_info=True)
                    skipped_count += 1
                    skipped_reasons.append(f"Bài {article_index}: Lỗi tải ({type(exc).__name__})")
                
                processed_futures += 1
                status_message = f"Đang tải bài: {processed_futures}/{total_entries_to_process}"
                if progress_callback: progress_callback(processed_futures, total_entries_to_process * 2, status_message)

        logging.info(f"Hoàn tất tải {len(fetched_articles_data)} bài. Bỏ qua {skipped_count} bài.")

        # --- 3. Classify and Summarize Fetched Articles --- 
        total_fetched = len(fetched_articles_data)
        processed_in_batching = 0
        status_message = f"Đang xử lý {total_fetched} bài đã tải..."
        if progress_callback: progress_callback(total_entries_to_process + processed_in_batching, total_entries_to_process * 2, status_message)

        # Process in batches for summarization efficiency
        batch_size = 3 # Adjust batch size based on GPU memory / performance
        all_summaries = [] # Store all summaries generated
        articles_for_summary = [] # Store corresponding article info

        # First, classify all fetched articles (can also be batched if model supports it well)
        classified_articles = []
        for data in fetched_articles_data:
            original_info = data["original_info"]
            title = original_info["title"]
            description = original_info["description"]
            article_index = original_info["index"]
            
            category = self.classify_article(title, description)
            if category == "Khác" and not description: # Simple retry with title only if description was empty
                 category = self.classify_article(title, "")
                 
            classified_articles.append({
                "content": data["content"],
                "category": category,
                "original_info": original_info
            })
            processed_in_batching += 1
            status_message = f"Đã phân loại: {processed_in_batching}/{total_fetched}"
            # Update progress after classification step (optional, can be slow)
            # if progress_callback: progress_callback(total_entries_to_process + processed_in_batching, total_entries_to_process * 2, status_message)

        logging.info(f"Hoàn tất phân loại {len(classified_articles)} bài.")
        processed_in_batching = 0 # Reset for summarization progress

        # Now, summarize in batches
        for i in range(0, len(classified_articles), batch_size):
            batch_data = classified_articles[i:min(i + batch_size, len(classified_articles))]
            batch_contents = [item["content"] for item in batch_data]
            batch_info = [item["original_info"] for item in batch_data]
            batch_categories = [item["category"] for item in batch_data]
            
            if not batch_contents:
                continue

            start_time_summary = time.time()
            try:
                # Adjust summarization parameters
                summaries = self.summarizer(
                    batch_contents,
                    max_length=160,  # Increased max length
                    min_length=50,   # Increased min length
                    do_sample=False, # Keep False for more deterministic summaries
                    batch_size=len(batch_contents) # Process the actual batch size
                )
                summaries_text = [s['summary_text'] for s in summaries]
                logging.info(f"Tóm tắt batch {i//batch_size + 1} ({len(summaries_text)} bài). Thời gian: {time.time() - start_time_summary:.2f}s")

                # Combine results
                for idx, summary in enumerate(summaries_text):
                    article_info = batch_info[idx]
                    category = batch_categories[idx]
                    news_item = {
                        "title": article_info["title"],
                        "link": article_info["link"],
                        "summary": summary,
                        "published": article_info["published"],
                        "category": category # Add category here
                    }
                    # Filter by selected category if not "Tất cả"
                    if selected_category == "Tất cả" or category == selected_category:
                         news_by_category[category].append(news_item)
                         
                    processed_count += 1 # Count all successfully summarized articles
            
            except PipelineException as e:
                logging.error(f"Lỗi Pipeline khi tóm tắt batch bắt đầu từ bài {i}: {e}", exc_info=True)
                skipped_count += len(batch_contents)
                error_indices = [info['index'] for info in batch_info]
                skipped_reasons.append(f"Bài {error_indices}: Lỗi Pipeline tóm tắt")
            except Exception as e:
                logging.error(f"Lỗi không xác định khi tóm tắt batch bắt đầu từ bài {i}: {e}", exc_info=True)
                skipped_count += len(batch_contents)
                error_indices = [info['index'] for info in batch_info]
                skipped_reasons.append(f"Bài {error_indices}: Lỗi không xác định tóm tắt")
            finally:
                 processed_in_batching += len(batch_contents)
                 status_message = f"Đã xử lý tóm tắt: {processed_in_batching}/{total_fetched}"
                 if progress_callback: progress_callback(total_entries_to_process + processed_in_batching, total_entries_to_process * 2, status_message)

        # --- 4. Finalize and Report --- 
        logging.info(f"Hoàn tất xử lý: {processed_count} bài được tóm tắt thành công, {skipped_count} bài bị bỏ qua.")

        # Create summary text for Telegram
        summary_text_tg = f"📰 Tóm tắt tin tức: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {region} | {selected_category} (Giới hạn: {article_limit_str})
(Đã xử lý: {processed_count}, Bỏ qua: {skipped_count})
====================

"
        category_counts = {cat: len(articles) for cat, articles in news_by_category.items() if articles}
        if not category_counts:
             summary_text_tg += "Không có tin tức nào được xử lý thành công."
        else:
            for category, articles in news_by_category.items():
                if articles:
                    summary_text_tg += f"📌 **{category}** ({len(articles)} bài):
"
                    for i, a in enumerate(articles[:5], 1): # Limit to 5 per category for TG message
                        pub_time_str = ""
                        try:
                             # Attempt to parse and format published time
                             pub_dt = datetime.fromisoformat(a['published'].replace("Z", "+00:00"))
                             pub_time_str = f" ({pub_dt.strftime('%d/%m %H:%M')})" 
                        except:
                             pub_time_str = f" ({a['published']})"
                             
                        summary_text_tg += f"{i}. {a['title']}{pub_time_str}
   - Tóm tắt: {a['summary']}
   - Link: {a['link']}

"
                    summary_text_tg += "--------------------
"
        # Send to Telegram (asynchronously)
        if self.bot:
            logging.info("Chuẩn bị gửi tóm tắt qua Telegram...")
            try:
                # Run the async function in the current event loop if available, or create a new one
                loop = asyncio.get_event_loop()
                if loop.is_running():
                     # If loop is running (like in Jupyter), create a task
                     # This might be complex in Streamlit context, consider running sync or dedicated thread
                     # For simplicity, try running it directly if no loop exists
                     logging.warning("Event loop is running. Sending Telegram message might block if not handled carefully.")
                     # Schedule the task without awaiting here in sync function
                     asyncio.create_task(self.send_to_telegram(summary_text_tg))
                     telegram_report.append("Gửi Telegram đã được lên lịch (không chặn).")
                else:
                     tg_error = asyncio.run(self.send_to_telegram(summary_text_tg))
                     if tg_error:
                          telegram_report.append(f"Lỗi gửi Telegram: {tg_error}")
                     else:
                          telegram_report.append("Đã gửi Telegram thành công.")
            except RuntimeError as e:
                 # Handle cases where no event loop exists or cannot be created/run
                 logging.error(f"RuntimeError khi gửi Telegram (có thể do môi trường không hỗ trợ asyncio.run): {e}")
                 # Try sending in a separate thread as a fallback (less ideal)
                 try:
                      thread = threading.Thread(target=lambda: asyncio.run(self.send_to_telegram(summary_text_tg)))
                      thread.start()
                      telegram_report.append("Gửi Telegram đã được lên lịch trong thread riêng (không chặn).")
                 except Exception as thread_e:
                      logging.error(f"Lỗi khi tạo thread để gửi Telegram: {thread_e}")
                      telegram_report.append(f"Lỗi nghiêm trọng khi cố gắng gửi Telegram: {e}")
            except Exception as e:
                 logging.error(f"Lỗi không xác định khi gửi Telegram: {e}", exc_info=True)
                 telegram_report.append(f"Lỗi không xác định khi gửi Telegram: {e}")
        else:
             telegram_report.append("Telegram Bot không được cấu hình.")

        # Final status message
        status_message = f"Hoàn tất: {processed_count} bài được xử lý, {skipped_count} bài bị bỏ qua."
        if skipped_count > 0 and skipped_reasons:
            status_message += f" (Lý do chính: {skipped_reasons[0]})"
        
        # Clean category dict for return (remove empty categories)
        final_news_by_category = {k: v for k, v in news_by_category.items() if v}

        return final_news_by_category, status_message, telegram_report

    def cleanup(self):
        """Dọn dẹp Selenium driver."""
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Đóng Selenium driver")
                self.driver = None
            except Exception as e:
                logging.error(f"Lỗi đóng Selenium driver: {e}", exc_info=True)

# Example usage (for testing, not part of the final Streamlit app)
if __name__ == '__main__':
    print("Testing NewsProcessor...")
    try:
        processor = NewsProcessor()
        print("Processor initialized.")
        
        # Define a simple progress callback for testing
        def test_progress(step, total_steps, message):
            print(f"Progress: [{step}/{total_steps}] {message}")

        # Run the processing (synchronously for testing)
        print("Starting news processing...")
        results, status, tg_report = processor.process_news(
            region="Việt Nam", 
            selected_category="Tất cả", 
            article_limit_str="5", 
            progress_callback=test_progress
        )
        print("Processing finished.")
        print(f"Status: {status}")
        print(f"Telegram Report: {tg_report}")
        
        if results:
            print("\n--- Results ---")
            for category, articles in results.items():
                print(f"\n## {category} ({len(articles)} articles)")
                for i, article in enumerate(articles[:2]): # Print first 2 articles per category
                    print(f"  {i+1}. {article['title']}")
                    print(f"     Summary: {article['summary']}")
                    print(f"     Link: {article['link']}")
        else:
            print("No results processed.")
            
    except ImportError as e:
         print(f"ERROR: Failed to initialize processor due to missing import: {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
         logging.error("Unexpected error during test execution", exc_info=True)
    finally:
        if 'processor' in locals() and processor:
            processor.cleanup()
            print("Processor cleaned up.")

