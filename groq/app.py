import time
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv('../.env')

groq_api = os.environ.get("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader("https://sahil.hadke.in/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("GROQ DEMO")
llm = ChatGroq(groq_api_key=groq_api, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retreiver = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retreiver, document_chain)

prompt = st.text_input("enter your prompt")

if prompt:
    output = retrieval_chain.invoke({'input': prompt})
    st.write(output['answer'])
