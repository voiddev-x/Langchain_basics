from langchain.tools import tool

@tool #Decorator 
def chiki(person: str) -> str :
    """Determines if the person is Chiki"""#Description
    if person.lower() == 'Ankit':
        return f'{person} is chiki'
    else:
        return f"{person} is not chiki"
    
    
print(chiki.invoke({'person' :'Vaibhav'}))
    
print(chiki.description)