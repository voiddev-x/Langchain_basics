from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

# load_dotenv(dotenv_path=Path("../.env"))

env_path = Path(__file__).parent.parent/ ".env"
load_dotenv(dotenv_path=env_path)

text_splitter = SemanticChunker(
    MistralAIEmbeddings(
        model="mistral-embed",
    ),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.7
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.

"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)

#DID NOT WORK PROPERLY; it's in experimental phase only does not really work  
