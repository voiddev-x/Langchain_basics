from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from typing import Literal
from pydantic import BaseModel,Field

load_dotenv()

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGroq(model="openai/gpt-oss-120b")
parser1 = StrOutputParser()


class sentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description = 'give the sentiment of the following text')
    
parser2 = PydanticOutputParser(pydantic_object=sentimentOutput) 
    
prompt1 = PromptTemplate(
    template = "analyze the feedback and classify the sentiment \n {feedback} \n{format_instructions} ",
    input_variables= ['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model2 | parser2

prompt2 = PromptTemplate(
    template = 'write an appropriate response for this positive feedback \n{feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = 'write an appropriate response for this negative feedback \n{feedback}',
    input_variables=['feedback']
)
branch_chain= RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':"this such cute but terrible phone"}))
