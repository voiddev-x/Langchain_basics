from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("../docs/GRU.pdf")

docs = docs.load()

print(docs[4].page_content)