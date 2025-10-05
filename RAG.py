import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# Load environment variables
load_dotenv()

# App settings
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="🎓 RAG — Egyptian Universities", layout="wide")
st.title("🎓 Egyptian Universities Chatbot")
st.markdown("Ask me anything about **public universities in Egypt!** 🇪🇬")

# Load Vector Database
st.info("📥 Loading stored knowledge from the Chroma Vector Database...")

# Markdown explanation beside data loading
st.markdown("""
> **ℹ️ About this step:**  
> The app is now connecting to the **Chroma Vector Database**,  
> which stores **scraped data** about Egyptian universities.  
> This database contains vector embeddings created from university information.  
> When you ask a question, the system retrieves the **most relevant documents**  
> to help the model generate an accurate and contextual answer.
""")

# Load the embeddings model and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# Retrieve top similar documents

def get_top_docs_text(db, query, k=3, char_limit_per_doc=500):
    docs = db.similarity_search(query, k=k)
    context_parts, sources = [], []
    for d in docs:
        text = getattr(d, "page_content", "")
        meta = getattr(d, "metadata", {})
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")
        excerpt = text[:char_limit_per_doc].strip()
        context_parts.append(f"{title}\n{excerpt}")
        sources.append({"title": title, "url": url})
    return "\n\n".join(context_parts), sources


# Generate answer using the LLM

def generate_answer(query, context):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "⚠️ Missing OPENROUTER_API_KEY in .env file."

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    prompt = f"""
    You are an assistant that answers questions about Egyptian public universities.

    Question: {query}

    Here is some context:
    {context}

    Answer clearly and concisely.
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
        return f"⚠️ Error generating answer: {e}"


# Streamlit UI

query = st.text_input("🔍 Ask your question:")
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            context, sources = get_top_docs_text(db, query)
            answer = generate_answer(query, context)

        st.success("✅ **Answer:**")
        st.write(answer)

        if sources:
            st.subheader("📚 Sources:")
            for i, s in enumerate(sources, start=1):
                st.markdown(f"{i}. [{s['title']}]({s['url']})")
