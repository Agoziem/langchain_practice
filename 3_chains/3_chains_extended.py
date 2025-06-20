from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model =  ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

def count_words_fn(text: str) -> str:
    return f"Word count: {len(text.split())}\n{text}"

def uppercase_fn(text: str) -> str:
    return text.upper()

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(count_words_fn)
uppercase_output = RunnableLambda(uppercase_fn)



# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)
