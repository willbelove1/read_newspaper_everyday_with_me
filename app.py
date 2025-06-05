
import streamlit as st
import os
import logging
from datetime import datetime

# Import the refactored processor class
from news_processor import NewsProcessor, REGIONS, CATEGORIES, LIMITS

# Configure logging (optional, Streamlit also has logging)
log_file = 'news_summary_streamlit.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8',
    errors='replace'
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Trình Tóm Tắt Tin Tức Nâng Cao",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Trình Tóm Tắt Tin Tức Nâng Cao")
st.caption(f"Cập nhật lần cuối: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Caching the NewsProcessor --- 
# Cache the processor initialization to avoid reloading models on every run
@st.cache_resource
def get_news_processor():
    try:
        processor = NewsProcessor()
        return processor
    except ImportError as e:
        st.error(f"Lỗi nghiêm trọng khi khởi tạo bộ xử lý: {e}. Vui lòng kiểm tra cài đặt thư viện (đặc biệt là sentencepiece) và thử lại.")
        logging.error(f"Failed to initialize processor: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Lỗi không xác định khi khởi tạo bộ xử lý: {e}")
        logging.error(f"Unknown error during processor initialization: {e}", exc_info=True)
        return None

processor = get_news_processor()

# --- Sidebar for User Input ---
st.sidebar.header("Cấu hình")

if processor:
    # Get default values (can be loaded from a config file or set directly)
    # For simplicity, using fixed defaults here
    default_region_index = list(REGIONS.keys()).index("Việt Nam") if "Việt Nam" in REGIONS else 0
    default_limit_index = LIMITS.index("5") if "5" in LIMITS else 0

    selected_region = st.sidebar.selectbox(
        "Chọn Khu Vực (Google News):",
        options=list(REGIONS.keys()),
        index=default_region_index,
        key="region_select"
    )

    selected_category = st.sidebar.selectbox(
        "Chọn Danh Mục:",
        options=["Tất cả"] + CATEGORIES,
        index=0, # Default to "Tất cả"
        key="category_select"
    )

    selected_limit = st.sidebar.selectbox(
        "Số Lượng Bài Tối Đa:",
        options=LIMITS,
        index=default_limit_index,
        key="limit_select"
    )

    # Button to trigger the process
    run_button = st.sidebar.button("🚀 Lấy và Tóm Tắt Tin Tức", key="run_button", use_container_width=True)

else:
    st.sidebar.warning("Bộ xử lý tin tức chưa được khởi tạo thành công. Không thể chạy.")
    run_button = False # Disable button if processor failed

# --- Main Area for Output ---
status_placeholder = st.empty() # Placeholder for status messages
progress_bar = st.progress(0) # Progress bar
results_area = st.container() # Container for displaying results

# --- Processing Logic --- 
if run_button and processor:
    results_area.empty() # Clear previous results
    status_placeholder.info(f"⏳ Bắt đầu quá trình lấy tin tức cho khu vực '{selected_region}', danh mục '{selected_category}', giới hạn '{selected_limit}'...")
    logging.info(f"User triggered run: region={selected_region}, category={selected_category}, limit={selected_limit}")

    # Define the progress callback for Streamlit
    def streamlit_progress_callback(step, total_steps, message):
        progress_percentage = 0
        if total_steps > 0:
            progress_percentage = min(int((step / total_steps) * 100), 100)
        progress_bar.progress(progress_percentage)
        status_placeholder.info(f"⏳ {message} ({step}/{total_steps})")
        logging.debug(f"Progress Update: [{step}/{total_steps}] {message}")

    try:
        start_process_time = time.time()
        # Call the processor's main function
        news_results, final_status, telegram_status = processor.process_news(
            region=selected_region,
            selected_category=selected_category,
            article_limit_str=selected_limit,
            progress_callback=streamlit_progress_callback
        )
        end_process_time = time.time()
        processing_time = end_process_time - start_process_time
        logging.info(f"Processing finished in {processing_time:.2f} seconds.")

        # Display final status
        status_placeholder.success(f"✅ {final_status} (Thời gian xử lý: {processing_time:.2f} giây)")
        progress_bar.progress(100)

        # Display Telegram status
        if telegram_status:
            for msg in telegram_status:
                if "Lỗi" in msg:
                    st.warning(f"📢 Telegram: {msg}")
                else:
                    st.info(f"📢 Telegram: {msg}")

        # Display results
        if news_results:
            results_area.subheader("Kết quả Tóm tắt Tin tức")
            # Sort categories for consistent display (optional)
            sorted_categories = sorted(news_results.keys(), key=lambda x: CATEGORIES.index(x) if x in CATEGORIES else float('inf'))
            
            for category in sorted_categories:
                articles = news_results[category]
                if articles:
                    with results_area.expander(f"**{category}** ({len(articles)} bài)", expanded=True):
                        for i, article in enumerate(articles):
                            st.markdown(f"**{i+1}. {article['title']}**")
                            # Attempt to parse and display published time nicely
                            try:
                                pub_dt = datetime.fromisoformat(article['published'].replace("Z", "+00:00"))
                                st.caption(f"Xuất bản: {pub_dt.strftime('%d/%m/%Y %H:%M:%S')}")
                            except:
                                st.caption(f"Xuất bản: {article['published']}") # Fallback to raw string
                            
                            st.markdown(f"**Tóm tắt:** {article['summary']}")
                            st.markdown(f"[Đọc bài gốc]({article['link']})", unsafe_allow_html=True)
                            st.divider()
        else:
            results_area.info("Không có kết quả nào để hiển thị.")

    except Exception as e:
        status_placeholder.error(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")
        progress_bar.progress(0)
        logging.error(f"Error during Streamlit process execution: {e}", exc_info=True)

# --- Add Footer or other info ---
st.sidebar.markdown("--- ")
st.sidebar.caption("Ứng dụng được phát triển bởi Manus.")

# Note: Cleanup of the processor (like closing Selenium) 
# should ideally happen when the Streamlit app stops.
# Streamlit's lifecycle makes this hard to guarantee perfectly.
# Using @st.cache_resource helps keep the processor alive between runs.
# Explicit cleanup might require more complex session state management or 
# running Streamlit with a wrapper script.

