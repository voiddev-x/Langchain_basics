from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

class person(BaseModel):
    name: str = Field(description= "name of the person")
    age: int = Field(description= "age of the person")
    city: str = Field(description= "name of the city person belongs to")
    
    
parser = PydanticOutputParser(pydantic_object=person)

template= PromptTemplate(
    template="Name age and city of an imaginary person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template| model | parser

result = chain.invoke({})
print(result)