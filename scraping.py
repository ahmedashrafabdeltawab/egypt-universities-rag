import os
import json
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def fetch_university_links():
    """
    Scrapes university links from UniversitiesEgypt.com (public universities page)
    """
    print("🚀 Starting scraping process...")
    base_url = "https://www.universitiesegypt.com"
    list_url = f"{base_url}/top-public-universities-in-egypt"

    try:
        resp = requests.get(list_url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch universities list: {e}")
        raise SystemExit("⚠️ Stopping scraping because university list could not be fetched.")

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    # Look for all anchor tags linking to university pages
    for a in soup.select("a[href^='/university/']"):
        href = base_url + a["href"]
        title = a.text.strip()
        if title and href not in [l["url"] for l in links]:
            links.append({"title": title, "url": href})

    if not links:
        raise ValueError("❌ No university links found on the page. Please check the website structure.")
    else:
        print(f"✅ Found {len(links)} university links.")
    return links


def scrape_universities_pages(universities):
    """
    Visits each university page and scrapes its content.
    """
    data = []
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    for i, uni in enumerate(universities, start=1):
        try:
            url = uni["url"].replace("//university", "/university") 
            print(f"📄 Scraping {i}/{len(universities)}: {url}")
            driver.get(url)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            paragraphs = " ".join([p.text.strip() for p in soup.find_all("p") if p.text.strip()])

            uni_data = {
                "name": uni["title"],
                "url": url,
                "content": paragraphs or "No detailed content found."
            }
            data.append(uni_data)

        except Exception as e:
            print(f"⚠️ Error scraping {uni['title']}: {e}")

    driver.quit()
    print(f"✅ Successfully scraped {len(data)} universities.")
    return data


def build_vector_database():
    """
    Converts scraped university data into vector embeddings and stores in ChromaDB.
    """
    print("\n🧠 Building Chroma Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir):
        print("✅ Chroma DB already exists. Loading it...")
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        with open("universities.json", "r", encoding="utf-8") as f:
            universities = json.load(f)

        documents = [
            Document(page_content=uni["content"], metadata={"source": uni["url"], "name": uni["name"]})
            for uni in universities
        ]

        db = Chroma.from_documents(documents, embedding_function=embeddings, persist_directory=persist_dir)
        db.persist()

    print("✅ Done! Vector DB is ready.")
    return db


if __name__ == "__main__":
    universities = fetch_university_links()
    scraped_data = scrape_universities_pages(universities)

    with open("universities.json", "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(scraped_data)} universities to universities.json")

    build_vector_database()
