import boto3
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings, ChatBedrock
#from langchain_community.chat_models import BedrockChat

from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# Define vectorstore
global vectorstore_faiss

# Define 3 convenience functions
def config_llm():
    client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-2")
    model_kwargs = {
        "max_tokens" : 512,
        "temperature" : 0.1,
        "top_p" : 1
    }
    model_id ="anthropic.claude-3-haiku-20240307-v1:0"
    #model_id ="anthropic.claude-3-sonnet-20240229-v1:0"
    #model_id ="anthropic.claude-3-7-sonnet-20250219-v1:0"
    llm = ChatBedrock(model_id=model_id, client=client, model_kwargs=model_kwargs)
    return llm

def config_vector_db(filename):
    client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-2") 
    bedrock_embeddings = BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v2:0")
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

def vector_search(query):
    global vectorstore_faiss  
    docs = vectorstore_faiss.similarity_search_with_score(query, k=3)  # Limit results
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
    Include all relevant information but keep your answer short. Only answer the question.
    Do not say things like "according to the training or handbook or according to the information provided..."

    <Information>
    {info}
    <Information>

    {input}

Assistant:
"""

#Configure prompt template
prompt_template= PromptTemplate(
    input_variables= ['input', 'info'],
    template= my_template
)

#Create llm chain
question_chain= LLMChain(
    llm= llm,
    prompt= prompt_template,
    output_key= 'answer'
)

#Get question, perform similarity search, invoke model and return result.
while True:
    question= input("\nWhat would like to learn from the By Miles Policy Handbook Manual ?\n")

    info= vector_search(question) # perform a similarity search
    output= question_chain.invoke({'input': question, 'info': info})  # invoke the model, providing additional context

    #display the result
    print(output['answer'])