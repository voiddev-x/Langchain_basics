from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path= env_path)

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(docs, embeddings)

s_retriever = vectorstore.as_retriever(search_kwarg = {'k':3})

print("s_search Result")

s_docs = s_retriever.invoke('what is gradient descent')

for doc in s_docs:
    print(doc.page_content)
    
    
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)
print('MMR Result')

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)

