from langchain_community.document_loaders import TextLoader
# import os 
docs = TextLoader("../docs/notes.txt")

docs = docs.load()
# print(os.getcwdb())
print(docs[0].page_content)