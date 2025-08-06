from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

def get_cypher_generate_model():
    llm = ChatOpenAI(model="gpt-4.1-nano")

    return llm