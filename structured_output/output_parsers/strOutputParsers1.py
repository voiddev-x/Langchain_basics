from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

#1st ==> detailed report
template1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables= ['topic']
)
#2nd ==> summary of the detailed report
template2 = PromptTemplate(
    template = 'write a summary of the following text in 30 words. /n {text}',
    input_variables= ['text']
) 

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black Hole"})

print(result)
