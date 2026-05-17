from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()


prompt = PromptTemplate(
    template='write a report on the following topic\n {topic}',
    input_variables=['topic']
)

chain1 = RunnableSequence(prompt, model , parser)
prompt1 = PromptTemplate(
    template = 'summarize the following report \n {report}',
    input_variables=['report']
)

chain2 = RunnableBranch(
    (lambda x: len(x.split())>500 , prompt1| model | parser ),
    RunnableLambda(lambda x: x)# should have used RunnablePassthrough
)

chain = RunnableSequence(chain1, chain2)
print(chain.invoke({'topic':'sleep'}))