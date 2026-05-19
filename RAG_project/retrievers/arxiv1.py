#Right now can't use arxiv 
# as the newest version is not supported by langchain(as it has removed .result) 
# and older version uses http and can't be redirected to https

from langchain_community.retrievers import PubMedRetriever

retriever = PubMedRetriever(
    top_k_results=3,
    load_all_available_meta=True
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"\n Result {i+1}")
    print("Title:",doc.metadata.get("Title"))
    print('summary:',doc.page_content[:500])