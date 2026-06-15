#same agent again using langchain's buit in create_agent fuction
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

@tool
def get_weather(city: str) -> str:
    """get the current weather of the city
    use this when you need to get the current weather conditions of a city 
    
    Args:
        city: The name of the city to get weather for(eg. 'pune' , 'hyderabad')
        
    Returns:
        A string describing the current weather conditions.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")    
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric" 
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }
        return (
                f"Weather in {weather_info['city']}, {weather_info['country']}:\n"
                f"  Temperature: {weather_info['temperature']}°C (feels like {weather_info['feels_like']}°C)\n"
                f"  Condition: {weather_info['description'].capitalize()}\n"
                f"  Humidity: {weather_info['humidity']}%\n"
                f"  Wind Speed: {weather_info['wind_speed']} m/s"
            )

        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"City,{city} not found. Please check the city name."
        return f"HTTP Error: {e}"
    
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool    
def get_news(city: str) -> str:
    """get the news about the city 
    use this tool when you have to get the news about a city

    Args:
        city (str): name of the city for which news is to obtained 

    Returns:
        str: news about the city 
    """
    
    search_tool = TavilySearch(max_results = 5, topic= "news")
    news = search_tool.invoke({"query": f"latest news in the {city}"})
    
    news_list = [
        {
            "title": article["title"],
            "url":article["url"],
            "snippet": article["content"][:150]
        }
        for article in news['results']
    ]
    return news_list
    
    
llm = ChatMistralAI(model_name="mistral-small-2506")

tools = [get_news, get_weather]
tool_registry = {t.name : t for t in tools}

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before every tool call"""
    tool_name = request.tool_call['name']
    confirm = input (f"Agent wants to cal '{tool_name}'. do you approve (y/n): ")
    
    if confirm.lower() != "y":
        return ToolMessage(
            content = 'tool call denied by user . Answer according to existing knowledge',
            tool_call_id = request.tool_call['id']
        )
        
    return handler(request)
    
    
agent = create_agent(
    llm,
    tools= tools,
    system_prompt="you are a helpful city guide (assistant)",
    middleware=[human_approval]
    
)

extract_result = RunnableLambda(lambda result : result["messages"][-1])
chain = agent| extract_result| StrOutputParser()

print("City Agent | type exit to exit |")



while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    result = chain.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    
    print("bot: ", result)