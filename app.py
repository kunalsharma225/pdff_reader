import streamlit as st
import fitz
import re
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
def clean_text(raw_text):
    return re.sub(r'[^\w\s.,;:!?()-]', '', raw_text)

def extract_text(pdf_file):
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text("text")
    pdf_doc.close()
    return clean_text(text)

def create_faiss_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def create_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    prompt_template = """You are an expert assistant.
Use the following context to answer the question concisely.
If you don't know, just say you don't know.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 Ask Questions from Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_file)
        db = create_faiss_db(text)
        qa_chain = create_qa_chain(db)
    st.success("✅ PDF processed! Ask your question below.")

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Getting answer..."):
            result = qa_chain.run(question)
            st.markdown("### 💬 Answer:")
            st.write(result)
