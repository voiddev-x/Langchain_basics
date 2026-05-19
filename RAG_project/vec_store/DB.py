from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document

env_path = Path(__file__).parent.parent/'.env'
load_dotenv(dotenv_path= env_path)

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory= "chroma_db"
)

result = vectorstore.similarity_search("what is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    
retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":2})

stack = retriever.invoke("what is used for data analysis?")

for s in stack:
    print(s.page_content)
    
    
    

# chroma_db Folder Structure Explained
# This is Chroma's local persistence layer — it saves your vectorstore to disk so you don't have to re-embed every time.

# The UUID Folder — a55623df-c443-...
# This folder represents a collection inside Chroma. The name is auto-generated. Think of a collection like a table in a database — it holds all your embedded documents together.

# The .bin Files (inside the UUID folder)
# These are HNSW index files (Hierarchical Navigable Small World) — the actual vector search engine under the hood.
# File &What it stores 
# data_level0.bin  The actual vector embeddings + neighbor connections at the base layer — the core of the search 
# indexheader.bin  Metadata about the index — dimensions, distance method (cosine/L2), number of vectors, etc.
# length.bin The byte lengths of each vector entry, used for navigating data_level0.bin 
# link_lists.bin Upper layer graph connections for the HNSW index — used for fast approximate nearest neighbor search
# Together these 4 files form the HNSW graph that makes similarity search fast.

# chroma.sqlite3
# This is the metadata database. It stores:

# Your page_content (the actual text)
# Your metadata (like {"source": "AI_book"})
# Internal IDs linking each document to its vector in the .bin files