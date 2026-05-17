from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables=['topic']
)
prompt1 = PromptTemplate(
    template = 'explain the joke given {joke}',
    input_variables = ['joke']
)

chain = RunnableSequence(prompt, model , parser)

chain1 = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': prompt1 | model |parser
})
chain2 = RunnableSequence(chain, chain1)

print(chain2.invoke({
    'topic':'ai'
}))