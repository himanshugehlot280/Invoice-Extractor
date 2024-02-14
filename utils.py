from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
import time
from pinecone import Pinecone

import pandas as pd
import ast
import re
import json

from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datetime import date
from langchain.chains import LLMChain


# Extract text from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to create Langchain Documents from user-uploaded PDF files
# Iterating over files in that user uploaded (PDF files), converting the RAW text (\n\0) into Lanchain Documents
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        pdf_data=get_pdf_text(filename)
        docs.append(Document(
            page_content=pdf_data,
            metadata={"name": filename.name,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))
    return docs

#Create embeddings instance
def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()
    return embeddings

#Function to push embedded data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
        )
    index=PineconeStore.from_documents(docs,embeddings,index_name=pinecone_index_name)
    
# Function to pull data from Pinecone Vector Store
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    print("20secs delay...")
    time.sleep(20)
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
    )
    index_name = pinecone_index_name
    index = PineconeStore.from_existing_index(index_name, embeddings)
    return index

# Function to query Pinecone Vector Store for similar documents
def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    pinecone = Pinecone(
        api_key=pinecone_apikey,environment=pinecone_environment
    )
    index_name = pinecone_index_name
    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k))

    return similar_docs
    



# Function to extract data from a given format using OpenAI language model
def extract_data(pages_data):
    template = """Extract ONLY the following values: 
    Customer Name,Address,City,State,Country,Pin,MobileNo,DOB,Email,PAN No,Insurance Company name,PolicyType,Product Name,Vehicle Registration Status,Vehicle No,Make,Model,Variant,Date of Registration,Year of Manufacturing,Type of Vehicle/OwnershipType,Vehicle Class,Vehicle Sub Class,ChasisNo,EngineNo,CC,Fuel,RTO,Zone,NCB,ODD,PCV/GCV/Misc/TW,Passenger/GVW,Bus Proposal Date,Policy Start Date,Policy Expiry Date,Policy No,Policy Issue Date,Business Type (New/Renwal/Rollover),Sum Insured,OD Net Premium,OwnerDriver(LPD),Roadside Assistance(WithoutBrokerage),GST/TaxAmount,Stamp Duty,Gross Premium,Payment Mode,Tran No,Tran Dated,BankName,Premium Receipt No,Prev Policy_no,Insured/Proposer Name and Without NilDep from this data: {pages}

    Format the extracted output as JSON with the following keys only: 
    Customer Name,Address,City,State,Country,Pin,MobileNo,DOB,Email,PAN No,Insurance Company name,PolicyType,Product Name,Vehicle Registration Status,Vehicle No,Make,Model,Variant,Date of Registration,Year of Manufacturing,Type of Vehicle/OwnershipType,Vehicle Class,Vehicle Sub Class,ChasisNo,EngineNo,CC,Fuel,RTO,Zone,NCB,ODD,PCV/GCV/Misc/TW,Passenger/GVW,Bus Proposal Date,Policy Start Date,Policy Expiry Date,Policy No,Policy Issue Date,Business Type (New/Renwal/Rollover),Sum Insured,OD Net Premium,OwnerDriver(LPD),Roadside Assistance(WithoutBrokerage),GST/TaxAmount,Stamp Duty,Gross Premium,Payment Mode,Tran No,Tran Dated,BankName,Premium Receipt No,Prev Policy_no,Insured/Proposer Name,Without NilDep
    """

    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    
     # Create an instance of the OpenAI language model
    llm = OpenAI(temperature=0,max_tokens=1000)

    # Split the input data into smaller pdf_data
    chunk_size = 4000
    pdf_data = [pages_data[i:i+chunk_size] for i in range(0, len(pages_data), chunk_size)]
    
    dicts = []
    for chunk in pdf_data:
        # Use the language model to extract data based on the templat
        str_dict = llm(prompt_template.format(pages=chunk))
        print(llm(prompt_template.format(pages=chunk)))
        # Convert the extracted data from a string to a dictionary
        dictionary = json.loads(str_dict)
        dicts.append(dictionary)

    combined_dict = {}
    # Combine dictionaries into one
    for d in dicts:
        for key, value in d.items():
            if key in combined_dict and combined_dict[key] == "NA":
                combined_dict[key] = value
            elif key not in combined_dict:
                combined_dict[key] = value

    return combined_dict

# Function to convert extracted data into a DataFrame
def to_df(data):
    headers = [
    "Customer Name",
    "Address",
    "City",
    "State",
    "Country",
    "Pin",
    "MobileNo",
    "DOB",
    "Email",
    "PAN No",
    "Insurance Company name",
    "PolicyType",
    "Product Name",
    "Vehicle Registration Status",
    "Vehicle No",
    "Make",
    "Model",
    "Variant",
    "Date of Registration",
    "Year of Manufacturing",
    "Type of Vehicle/OwnershipType",
    "Vehicle Class",
    "Vehicle Sub Class",
    "ChasisNo",
    "EngineNo",
    "CC",
    "Fuel",
    "RTO",
    "Zone",
    "NCB",
    "ODD",
    "PCV/GCV/Misc/TW",
    "Passenger/GVW",
    "Bus Proposal Date",
    "Policy Start Date",
    "Policy Expiry Date",
    "Policy No",
    "Policy Issue Date",
    "Business Type (New/Renwal/Rollover)",
    "Sum Insured",
    "OD Net Premium",
    "OwnerDriver(LPD)",
    "Roadside Assistance(WithoutBrokerage)",
    "GST/TaxAmount",
    "Stamp Duty",
    "Gross Premium",
    "Payment Mode",
    "Tran No",
    "Tran Dated",
    "BankName",
    "Premium Receipt No",
    "Prev Policy_no",
    "Insured/Proposer Name",
    "Without NilDep"
]
    # Ensure all headers are present in the data; if not, set to 'NA
    for header in headers:
        if header not in data:
            data[header] = 'NA'
    
    # Create a DataFrame with the specified headers
    df = pd.DataFrame(data, index=[0])
    df = df[headers]
    return df