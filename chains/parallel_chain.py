from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from following text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short question answer based on the following text \n {text}',
    input_variable = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into one document \n {notes} and {quiz}',
    input_variables= ['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    "notes":prompt1 | model1 | parser,
    "quiz":prompt2| model2 |parser 
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain


#text from "Homo Deus"
text = """
The Black Death was not a singular event, nor even the worst plague in history. More disastrous
epidemics struck America, Australia and the Pacific Islands following the arrival of the first
Europeans. Unbeknown to the explorers and settlers, they brought with them new infectious diseases
against which the natives had no immunity. Up to 90 per cent of the local populations died as a
result.
7
On 5 March 1520 a small Spanish flotilla left the island of Cuba on its way to Mexico. The ships
carried 900 Spanish soldiers along with horses, firearms and a few African slaves. One of the slaves,
Francisco de Eguía, carried on his person a far deadlier cargo. Francisco didn’t know it, but
somewhere among his trillions of cells a biological time bomb was ticking: the smallpox virus. After
Francisco landed in Mexico the virus began to multiply exponentially within his body, eventually
bursting out all over his skin in a terrible rash. The feverish Francisco was taken to bed in the house
of a Native American family in the town of Cempoallan. He infected the family members, who
infected the neighbours. Within ten days Cempoallan became a graveyard. Refugees spread the
disease from Cempoallan to the nearby towns. As town after town succumbed to the plague, new
waves of terrified refugees carried the disease throughout Mexico and beyond.
The Mayas in the Yucatán Peninsula believed that three evil gods – Ekpetz, Uzannkak and Sojakak
– were flying from village to village at night, infecting people with the disease. The Aztecs blamed it
on the gods Tezcatlipoca and Xipe, or perhaps on the black magic of the white people. Priests and
doctors were consulted. They advised prayers, cold baths, rubbing the body with bitumen and
smearing squashed black beetles on the sores. Nothing helped. Tens of thousands of corpses lay
rotting in the streets, without anyone daring to approach and bury them. Entire families perished
within a few days, and the authorities ordered that the houses were to be collapsed on top of the
bodies. In some settlements half the population died.
"""


result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()