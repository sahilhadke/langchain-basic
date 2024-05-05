from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st

load_dotenv('.env')

# Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot. You are chatting with a user."),
        ("user", "Question: {question}")
    ]
)

# streamlit framework
st.title("Langchain Demo")
input_text=st.text_input("Enter something to chat..")

# gemini llm
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
op_parser = StrOutputParser()
chain = prompt|llm|op_parser

if input_text:
    result = chain.invoke({"question": input_text})
    st.write(result)