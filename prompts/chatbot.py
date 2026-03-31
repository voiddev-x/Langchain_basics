from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage 
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
model = ChatMistralAI(model="mistral-small-latest")

chat_history = [
    SystemMessage(content = 'you are a helpful ai assistant'),
    SystemMessage(content = 'give all answers under 30 words')
]

while True:
    user_input = input("you: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage( content = result.content))
    print("AI: ",result.content)
    
print(chat_history)