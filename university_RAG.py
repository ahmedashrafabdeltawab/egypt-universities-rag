import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
import requests  
from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# App Configuration
st.set_page_config(page_title="RAG — Egyptian Universities", layout="wide")
st.title("🎓 RAG — Public Universities in Egypt")
st.markdown("Prototype: Ask questions about public universities in Egypt.")

BASE_URL = "https://www.universitiesegypt.com"
JSON_PATH = "universities.json"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()

# Scraping Functions
def load_fallback_data(): #داله احطياتي لو فشل الموقع
    """Fallback university data if scraping fails."""
    return [
        {"title": "Cairo University", "url": f"{BASE_URL}/university/cairo-university", "content": "Cairo University is one of Egypt’s oldest and most prestigious universities."},
        {"title": "Alexandria University", "url": f"{BASE_URL}/university/alexandria-university", "content": "Alexandria University offers a variety of undergraduate and graduate programs."},
        {"title": "Ain Shams University", "url": f"{BASE_URL}/university/ain-shams-university", "content": "Ain Shams University is known for its research and innovation in Egypt."},
    ]

def get_universities_links():
    """Get list of university URLs (fallback to known ones if website fails)."""
    url = f"{BASE_URL}/public-universities"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"} # عشان يتعامل كأنه متصفح حقيقي

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404: #لو غلط بيرجع لينكات الموقع التلاته 
            st.warning("⚠️ The website structure has changed — loading sample universities instead.")
            return [
                f"{BASE_URL}/university/cairo-university",
                f"{BASE_URL}/university/alexandria-university",
                f"{BASE_URL}/university/ain-shams-university"
            ]
        resp.raise_for_status() #بيوقف التنفيذ لو حصل خطأ
        soup = BeautifulSoup(resp.text, "lxml") 
        links = []
        for a in soup.select("a[href^='/university/']"): 
            href = a.get("href")
            if href and href not in links:
                full = f"{BASE_URL.rstrip('/')}{href}"
                links.append(full)
        return links if links else load_fallback_data() # لو مجمعش داتا يرجع للنسخه الاحتياطي اللي فوق 
    except requests.RequestException as e:
        st.warning(f"Failed to fetch {url}: {e}. Using fallback data.")
        return load_fallback_data()

def scrape_university_page(url, retries=3, delay=2): #بيحاول فتح المتصفح 3 مرات
    """Scrape a single university page using Selenium.""" 
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "lxml")
            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title"
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            content = " ".join(paragraphs)
            driver.quit()
            return {"url": url, "title": title, "content": content}
        except Exception as e:
            st.warning(f"Error fetching {url} (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    st.error(f"Skipping {url} after {retries} failed attempts")
    driver.quit()
    return None

def scrape_all_universities():
    """Scrape all universities or use fallback data."""
    links = get_universities_links()
    results = []
    progress = st.progress(0)
    for i, link in enumerate(links, start=1):
        uni = scrape_university_page(link)
        if uni:
            results.append(uni)
        progress.progress(i / len(links))
        time.sleep(0.2)
    
    if not results:
        st.warning("No data scraped. Using fallback data.")
        results = load_fallback_data()
    
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results

def convert_to_documents(data):
    """Convert scraped data to LangChain Documents."""
    docs = []
    for uni in data:
        text = f"Name: {uni.get('title', '')}\nURL: {uni.get('url', '')}\n\n{uni.get('content', '')}"
        metadata = {"title": uni.get("title", ""), "url": uni.get("url", "")}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

# Load or Build Vector Database
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = None

try:
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        st.info("Loading existing Chroma DB...")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        st.info("Scraping data from website (first run only)...")
        universities = scrape_all_universities()
        
        if not universities:
            st.error("No university data available to build Chroma DB.")
            raise ValueError("No university data scraped or loaded from fallback.")
        
        st.success(f"✅ Scraped {len(universities)} universities. Building Chroma DB...")
        docs = convert_to_documents(universities)
        db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
        db.persist() # بيخزن الداتا بشكل دائم 
        st.success("Chroma DB created and saved successfully!")
except Exception as e:
    st.error(f"Failed to setup Chroma DB: {e}")
    if os.path.exists(JSON_PATH):
        st.info("Loading fallback data from JSON file...")
        with open(JSON_PATH, "r") as f:
            universities = json.load(f)
        if universities:
            docs = convert_to_documents(universities)
            db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
            db.persist()
            st.success("Chroma DB created from fallback JSON data.")

# Get top documents from vector DB
def get_top_docs_text(db, query, k=3, char_limit_per_doc=500):
    """Retrieve top-k documents from Chroma DB."""
    if db is None:
        return "⚠️ No Chroma DB available due to initialization failure.", []
    docs = db.similarity_search(query, k=k)
    if not docs:
        return "", []

    context_parts = []
    sources = []

    for d in docs:
        text = getattr(d, "page_content", str(d))
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")
        excerpt = text[:char_limit_per_doc].strip()
        context_parts.append(f"{title}\n{excerpt}")
        sources.append({"title": title, "url": url})

    context = "\n\n".join(context_parts)
    return context, sources

# Generate answer using DeepSeek via OpenRouter
def generate_answer_with_openai(query, context):
    """Generate an answer using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "⚠️ Missing OPENROUTER_API_KEY in .env file."

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    prompt = f"""
You are an assistant that answers questions about Egyptian public universities.

Question: {query}

Here is some context information extracted from real websites:
{context}

Answer the question in a clear, concise way.
    """

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error while generating answer: {e}"

# Streamlit UI
st.subheader("Ask a Question")
query = st.text_input("Type your question here:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            context, sources = get_top_docs_text(db, query, k=3)
            answer = generate_answer_with_openai(query, context)

        st.success("✅ Answer:")
        st.write(answer)

        with st.expander("View Retrieved Context"):
            st.write(context)

        if sources:
            st.subheader("📚 Sources used:")
            for i, s in enumerate(sources, start=1):
                title = s.get("title", "Unknown")
                url = s.get("url", "")
                st.markdown(f"{i}. [{title}]({url})")
