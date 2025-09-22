import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# Define vectorstore
global vectorstore_faiss

# Define 3 convenience functions
def config_llm():
    client = boto3.client('bedrock-runtime')

    model_kwargs = {
        "max_tokens" : 512,
        "temperature" : 0.1,
        "top_p" : 1
    }

    modelId ="anthropic.claude-3-haiku-20240307-v1:0"
    #modelId ="anthropic.claude-3-sonnet-20240229-v1:0"
    #modelId ="anthropic.claude-3-7-sonnet-20250219-v1:0"
    llm = Bedrock(model_id=model_id, client=client)
    llm.model_kwargs = model_kwargs
    return llm

def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client)
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

def vector_search(query):
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info += doc[0].page_content+'\n'
    return info

# Configuring the LLM and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db("Policy_handbook.pdf")



# Creating template
my_template = """
Human:
    You are a conversational assistant designed to help answer questions from a customer.
    You should reply to the customer's question using the information provided below. 

    <Information>
    {info}
    <Information>

    {input}

Assistant:
"""

