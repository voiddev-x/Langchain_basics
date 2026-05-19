from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader('../docs/GRU.pdf')
docs = data.load()

splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 5
)

chunks = splitter.split_documents(docs)

print(chunks[10].page_content)
