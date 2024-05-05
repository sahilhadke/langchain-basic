from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv('.env')

llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("Write a ballad about LangChain")
print(result.content)