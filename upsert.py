#UPSERTS PDFs TO PINECONE VECTOR DATABASE
#Some of this code is not used...

import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import time

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

index_name = "closedai"
host = "https://closedai-e1e1c23.svc.aped-4627-b74a.pinecone.io"
region = "us-east-1"
cloud = "aws"


# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = st.secrets["PINECONE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

# configure client
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(cloud=cloud, region=region)

index = pc.Index(index_name)
# wait a moment for connection
time.sleep(1)

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

from langchain_community.document_loaders import PyPDFLoader

#Manual PDF loader
loader = PyPDFLoader("/users/scottdunphy/downloads/metlife-public-fixed-income.pdf")
documents = loader.load()

#Splitter of text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

#Split the documents into chunks
docs = text_splitter.split_documents(documents)

#Add metadata for better vector search results
for doc in docs:
    doc.metadata["Industry"] = "Public Fixed Income"
    doc.metadata["Company"] = "MetLife"

embeddings = OpenAIEmbeddings()

#Connect to Pinecone Vector DB
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#Upsert documents to Pinecone Vector DB
vectorstore.add_documents(docs)




