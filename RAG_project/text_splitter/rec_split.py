from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader('../docs/GRU.pdf')
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 5
)

chunks = splitter.split_documents(docs)

print(chunks[45])