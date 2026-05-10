from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
parser = JsonOutputParser()#does not enforce schema


#template
template = PromptTemplate(
    template = 'Give me 3 facts about{topic}  \n {format_instruction}',
    input_variables = ["topic"],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()

# print(prompt)

# result = model.invoke(prompt)

# # print(result)
# final_result = parser.parse(result.content)

chain = template | model |parser

final_result = chain.invoke({"topic":"black hole"})

print(final_result)
# print(type(final_result))