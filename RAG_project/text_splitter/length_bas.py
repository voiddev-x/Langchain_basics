from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data = TextLoader("../docs/notes.txt")
docs = data.load()

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 100,
    chunk_overlap = 5
)

chunks = splitter.split_documents(docs)

print(type(chunks))