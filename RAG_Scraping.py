import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI




# Config 
st.set_page_config(page_title="RAG — Egyptian Universities", layout="wide")
st.title("🎓 RAG — Public Universities in Egypt")
st.markdown("Prototype: Ask questions about public universities in Egypt. Follow the buttons in the sidebar step-by-step.")

BASE_URL = "https://www.universitiesegypt.com/"
JSON_PATH = "universities.json"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Scraping functions

def get_universities_links():
    url = f"{BASE_URL}/public-universities"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)

    if resp.status_code == 404:
        st.warning("⚠️ The website structure has changed — loading sample universities instead.")
        return [
            f"{BASE_URL}/university/cairo-university",
            f"{BASE_URL}/university/alexandria-university",
            f"{BASE_URL}/university/ain-shams-university"
        ]
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in soup.select("a[href^='/university/']"):
        href = a.get("href")
        if href:
            full = BASE_URL + href
            if full not in links:
                links.append(full)
    return links


def scrape_university_page(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        st.warning(f"Skipping (failed to fetch): {url}")
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title"
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    content = " ".join(paragraphs)
    return {"url": url, "title": title, "content": content}

def scrape_all_universities(progress_callback=None):
    links = get_universities_links()
    results = []
    for i, link in enumerate(links, start=1):
        uni = scrape_university_page(link)
        if uni:
            results.append(uni)
        if progress_callback:
            progress_callback(i, len(links))
        time.sleep(0.2)
    return results


# Persistence helpers

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# LangChain documents & Chroma

def convert_to_documents(data):
    docs = []
    for uni in data:
        text = f"Name: {uni.get('title','')}\nURL: {uni.get('url','')}\n\n{uni.get('content','')}"
        metadata = {"title": uni.get("title",""), "url": uni.get("url","")}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_chroma(persist_directory=CHROMA_DIR, model_name=EMBEDDING_MODEL):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db 

def get_top_docs_text(db, query, k=3, char_limit_per_doc=500):
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


# OpenAI answer generation

def generate_answer_with_openai(query, context):
    """
    Generate an answer using DeepSeek via OpenRouter API.
    Make sure OPENROUTER_API_KEY is in your .env file.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "⚠️ Missing OPENROUTER_API_KEY in .env file."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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

        answer = completion.choices[0].message.content.strip()
        return answer

    except Exception as e:
        return f"⚠️ Error while generating answer: {e}"


# Auto-load JSON data

if not os.path.exists(JSON_PATH):
    st.info("universities.json not found → scraping data from website...")
    universities = scrape_all_universities()
    save_json(JSON_PATH, universities)
    st.success(f"Scraped and saved {len(universities)} universities to {JSON_PATH}")
else:
    universities = load_json(JSON_PATH)
    st.success(f"Loaded {len(universities)} universities from {JSON_PATH}")


# Load or create Chroma DB

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        st.info("Loading existing Chroma DB...")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        st.info("Creating Chroma DB (building embeddings)...")
        docs = convert_to_documents(universities)
        db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
        db.persist()
        st.success("Chroma DB ready ✅")
except Exception as e:
    st.error(f"Failed to setup Chroma DB: {e}")


# Streamlit UI: Ask question

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

        # Show retrieved context
        with st.expander("View Retrieved Context"):
            st.write(context)

        # Show sources
        if sources:
            st.subheader("📚 Sources used:")
            for i, s in enumerate(sources, start=1):
                title = s.get("title", "Unknown")
                url = s.get("url", "")
                st.markdown(f"{i}. [{title}]({url})")
