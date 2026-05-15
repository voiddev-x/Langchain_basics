from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# print("success")

prompt = PromptTemplate(
    template= 'generate 5 interesting facts about {topic}',
    input_variable = ['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({"topic": "cricket"})

# print(result)

chain.get_graph().print_ascii()