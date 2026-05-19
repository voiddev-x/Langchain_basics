from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()
 
parser = StrOutputParser()
 
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory="chromaM_db",
    embedding_function= embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.7#default = 0.5
    }
)



prompt = ChatPromptTemplate.from_messages(
    [('system',"""
      you are a helpful AI assistant
      use only provided context to answer the question.
      
      if answer not present in context,
      say: answer not found in doc
      """),
     
     ('human',"""
      context:{context}
      
      Question:{Question}""")]
)
def join_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])
    
    
chain1 = RunnableParallel({
    'Question': RunnablePassthrough(),
    'context': retriever  | RunnableLambda(join_doc)
})
final_chain = chain1| prompt| model |parser

print("Rag system created")

print('press 0 to exit')

while True:
    query= input('You : ')
    if query == "0":
        break
    response = final_chain.invoke(query)
    print(f"\n Response:{response} \n")


