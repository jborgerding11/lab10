# Lab 10: Creating and using a RAG vector store to assist a chatbot
# Note: We are not creating a 'persistant' store, so each time the chatbot is launched, the vector store is recreated

import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
openai_model = 'gpt-4o-mini' 
client = OpenAI(api_key = api_key)
import ast

# The default plumbing of the Jupyter template is not entirely compatible with our use of chroma, 
# so the next several lines of code align the template and our packages
__import__('pysqlite3') #  module name determined at runtime (two commands below)
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') # identifies the correct (vs default) sqllite3 version
import chromadb

# Next, load our text file with our domain 'facts'
facts = "facts.txt"

def embed_facts(facts):
    """
    Embeds facts into a vector store
    
    Parameters: A string of domain facts
        
    Returns: The vector store with the facts embedded
    """
    string_of_facts = open(facts, "r")  
    data = string_of_facts.read() 
    list = ast.literal_eval(data)

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="my_collection") 
    collection.upsert( 
        documents=list,
        ids=["id1", "id2", "id3"] 
    )
    return collection

def chatgpt_response(question):
    """
    Receives user query, adds a system prompt to the query, then adds additional facts as retrieved from a vector store
    
    Parameters: Receives query from user retrieved via chatbot
        
    Returns: OpenAI response to query
    """

    system_content = "You are a professional landscaper and provide landscaping advice and " \
                     "expertise to residential home owners in United States Department of Agriculture (USDA) " \
                     "Plant Hardiness Zone 8. "

    results = collection.query(
         query_texts=[question], # Chroma will embed the query and then perform a similarity search
         n_results=1 # how many results to return, default is 10
     )
    returned_message = results['documents'][0][0]

    system_content = system_content + "Consider too the RAG retrieval for the user query returned: " + returned_message
    print(system_content)

    chat_completion = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": question},
                  {"role": "system", "content": system_content}]
    )
    return chat_completion.choices[0].message.content


collection = embed_facts(facts) # Call once, not for each query



interface = gr.Interface(
    inputs = gr.Textbox(label="How can we help you with your yard?"),
    fn = chatgpt_response,
    outputs = gr.Textbox(label="Our Reply"),
    title = "Landscaping Support for DFW",
    description = "Landscaping Support for DFW"
)

interface.launch()