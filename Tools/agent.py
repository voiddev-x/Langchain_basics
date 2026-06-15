from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from langchain_tavily import TavilySearch

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
    
    url =     url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"


    # params= { they are hardcoded in the url this time
    #     "q":city,
    #     "appid":api_key,
    #     "units":"metrics"
    # }
    
    
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
    
    # news_list = [] my thought 
    # for i in news['results']:
    #     news_article = []
    #     url = i['url']
    #     title = i['title']
    #     snippet = i['content'][:100]
    #     news_article.append(url)
    #     news_article.append(title)
    #     news_article.append(snippet)
    #     news_list.append(news_article)
        
    #got it optimized by ai --> looks much better is much cleaner
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

T_llm = llm.bind_tools(tools)

messages = []

print('City Intelligence System')
print("type Exit to quit")

while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    
    while True:
        result = T_llm.invoke(messages)
        messages.append(result)
        
        if result.tool_calls:
            
            for tool_call in result.tool_calls:
                
                tool_name = tool_call['name']
                
                #HUMAN IN THE LOOP 
                confirm = input (f"Agent wants to call {tool_name} do you approve (ys/no):  ")
                
                if confirm.lower()=="no":
                    
                    print("tool call denied and i cannot access the latest information ")
                    messages.append(ToolMessage(
                        content= "Tool call denied by user. Answer using your existing knowledge. ",
                        tool_call_id = tool_call["id"]
                    ))
                    continue
                
                tool_result = tool_registry[tool_name].invoke(tool_call)
                messages.append(ToolMessage(
                    content= tool_result,
                    tool_call_id = tool_call["id"]
                ))
                
            continue
        else:
            print(result.content)
            break
 
        
        
                        
    
         