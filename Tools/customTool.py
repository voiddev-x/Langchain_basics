
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from rich import print
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_text_length(text: str)-> int:
    """Returns the number of character in a given text"""
    return len(text)

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite")

tools = [get_text_length]#a list with all the tools 
tool_registry = {t.name : t for t in tools}

#tool binding
T_llm = llm.bind_tools(tools)

messages= []
user_query = input("How may i help you: ")
query = HumanMessage(user_query)
messages.append(query)

result = T_llm.invoke(messages)

messages.append(result)

if result.tool_calls:
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_message = tool_registry[tool_name].invoke(tool_call)
        messages.append(tool_message)
        
final_response = T_llm.invoke(messages)
print(final_response.content)

 
# tool_registry = {t.name : t for t in tools}
# t.name : t for t in tool is just a short form for t in tools: 
#tool_registry[t.name]= t and @tool add an .name directory to the function which is just the name of the function in string form
#so here name of the function is used as key and the actual function as values