import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# -------------------- LOAD ENV --------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

# -------------------- LLM & EMBEDDINGS --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------- UI --------------------
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_google.pkl"

main_placeholder = st.empty()

# -------------------- PROCESSING --------------------
if process_url_clicked:

    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    docs = []
    main_placeholder.text("Fetching Articles...")

    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ")

            if len(text.strip()) > 500:
                docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            st.error(f"Error loading {url}: {e}")

    st.write("Number of documents fetched:", len(docs))

    if len(docs) == 0:
        st.error("No valid content could be fetched from the URLs.")
        st.stop()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    # Add unique IDs to each document
    for i, doc in enumerate(split_docs):
        doc.id = str(i)

    st.write("Number of chunks created:", len(split_docs))

    # Build FAISS vector store
    main_placeholder.text("Building Vector Store...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    time.sleep(1)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing Complete ✅")
    st.success("Processing Complete ✅")

# -------------------- QUERY SECTION --------------------
query = st.text_input("Ask a Question about the Articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Get relevant docs
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)

        # Build context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata["source"] for doc in relevant_docs]))

        # Ask Gemini
        prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""
        response = llm.invoke(prompt)

        st.header("Answer")
        st.write(response.content)

        st.subheader("Sources:")
        for source in sources:
            st.write(source)

    else:
        st.warning("Please process URLs first.")