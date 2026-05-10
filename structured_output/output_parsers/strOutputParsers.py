from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

#1st ==> detailed report
template1 = PromptTemplate(
    template = 'write a detailed report on {topic}',
    input_variables= ['topic']
)
#2nd ==> summary of the detailed report
template2 = PromptTemplate(
    template = 'write a summary of the following text. /n {text}',
    input_variables= ['text']
) 
prompt1 = template1.invoke({'topic':'black holes'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result2 = model.invoke(prompt2)

print(result2.content)



