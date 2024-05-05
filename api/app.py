from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn

from dotenv import load_dotenv
import os

load_dotenv('../.env')

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(
    title="LangChain API",
    description="API for LangChain",
    version="0.1"
)

prompt = ChatPromptTemplate.from_template("write me an essay about {topic}")


add_routes(
    app,
    prompt|llm,
    path="/api",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
