from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-small-latest")

messages = [
    SystemMessage(content = 'you are a helpful assistant'),
    HumanMessage(content = "Tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
