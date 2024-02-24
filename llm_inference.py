
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    LlamaChatContentFormatter,
)
from langchain.prompts import PromptTemplate

from dotenv import dotenv_values
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
config = dotenv_values(".env") 

AZURE_LLM_URL = config["AZURE_LLM_URL"]
AZURE_API_KEY = config["AZURE_API_KEY"]

TEMPERATURE = 0.8
MAX_TOKENS = 400

def get_query_for_recipe(main_ingredient):
    QUERY = f"Generate a recipe with {main_ingredient} as the main ingredient"
    print(f"Query: {QUERY}")
    return QUERY

# Load the LLM model
llm_azure = AzureMLChatOnlineEndpoint(
    endpoint_url=AZURE_LLM_URL,
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key=AZURE_API_KEY,
    content_formatter=LlamaChatContentFormatter(),
    model_kwargs={  "temperature":      TEMPERATURE, 
                    "max_new_tokens":   MAX_TOKENS},
)

# Create a prompt with query as input parameter
prompt = PromptTemplate(
    template="Answer the user query.\n{query}\n",
    input_variables=["query"]
)
print(f"prompt: {prompt.template}")

# Create a chain with prompt and model
chain = prompt | llm_azure

# Invoke the chain for a query - generate a recipe with rice as the main ingredient
query = get_query_for_recipe(main_ingredient="rice")
answer = chain.invoke({"query": query})
print(answer.content)