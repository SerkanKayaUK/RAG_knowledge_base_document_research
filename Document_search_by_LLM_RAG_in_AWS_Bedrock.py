import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# Define vectorstore
global vectorstore_faiss

# Define convenience functions
def config_llm():
    