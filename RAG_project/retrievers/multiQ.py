from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
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

llm = ChatMistralAI(model = "ministral-3b-2512")

retriever = vectorstore.as_retriever()

multiQ_retriever = MultiQueryRetriever.from_llm(
    retriever =retriever,
    llm = llm
)

query = "What is gradient descent?"

docs = multiQ_retriever.invoke(query)

for doc in docs:
    print(doc.page_content)

