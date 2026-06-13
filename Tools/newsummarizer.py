#This is for learning how built in Tools work in LangChain 
#Here we will be using a built in tool to perform websearch

from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

search_tools = TavilySearch(max_result = 5)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """you are news expert who knows what section of the a news can be discarded what can not 
    so summarize the following news provided to you
    
    {news}
    """
)
def return_query(tavily_ouput: dict):
    return tavily_ouput['query']

chain = search_tools| RunnableLambda(return_query) | prompt | model | parser

result = chain.invoke({'query':"latest news related to development in RAG" })

print (result)