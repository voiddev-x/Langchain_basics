from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "extract 5 most important points from {report}",
    input_variables=['report']
)

chain = prompt1| model | parser | prompt2 | model | parser

result = chain.invoke({'topic': "swan"})
print(result)