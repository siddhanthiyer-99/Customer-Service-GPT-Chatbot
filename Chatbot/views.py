import openai
import os
import supabase
import pandas as pd
import psycopg2

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse

from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from .models import chatlog

from .utils import chat_response, subclass_create_complaint_handler, subclass_update_complaint_handler, subclass_view_complaint_handler
from simple_salesforce import Salesforce
from pymongo.collection import Collection
from pymongo import MongoClient
import logging
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

def salesforce_connect() -> Salesforce:
    """
    Initialize a Salesforce client object.

    This function attempts to establish a connection to Salesforce using the provided credentials
    fetched from environment variables: SF_USERNAME, SF_PASSWORD, and SF_SECURITY_TOKEN.

    Returns:
        Salesforce client object: An instance of Salesforce client upon successful connection.

    Raises:
        Exception: If any error occurs during the connection attempt, it catches the exception
                   and returns an error message string.
    """
    try: 
        sf = Salesforce(username=os.getenv('SF_USERNAME'), 
                        password=os.getenv('SF_PASSWORD'), 
                        security_token=os.getenv('SF_SECURITY_TOKEN'))
        return sf
    except Exception as e:
        return f"Error: {e}"
    
def mongodb_collection(database_name: str, collection_name: str) -> Collection:
    """
    Establishes a connection to a MongoDB collection using the provided database name and collection name.

    Parameters:
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the collection within the specified database.

    Returns:
        Collection: A MongoDB Collection object representing the specified collection.

    Raises:
        ConnectionError: If unable to establish a connection to the MongoDB database.
        
    Example:
        >>> collection = mongodb_collection('my_database', 'my_collection')
    """
    connection_string = os.getenv('ATLAS_CONNECTION_STRING')
    try:
        client = MongoClient(connection_string)
        client.server_info()
        chat_collection = client[database_name][collection_name]
        return chat_collection
    except Exception as e:
        logging.error('Database connection failure: {}'.format(e))
        return False

def vector_search(database_name: str, collection_name: str, index_name: str) -> MongoDBAtlasVectorSearch:
    """
    Initializes a MongoDB Atlas Vector Search instance for vector-based search operations.

    Parameters:
        database_name (str): The name of the MongoDB database containing the collection.
        collection_name (str): The name of the collection within the specified database.
        index_name (str): The name of the index to be used for vector search operations.

    Returns:
        MongoDBAtlasVectorSearch: An instance of MongoDB Atlas Vector Search configured for the specified collection and index.

    Example:
        >>> search_instance = vector_search('my_database', 'my_collection', 'my_index')
    """
    collection = mongodb_collection(database_name=database_name, collection_name=collection_name)
    embeddings_model_name = os.getenv('EMBEDDINGS_MODEL')
    vector_search = MongoDBAtlasVectorSearch(collection=collection,embedding=OpenAIEmbeddings(model=embeddings_model_name), index_name=index_name)
    return vector_search


system_prompt_template = SystemMessagePromptTemplate.from_template(
    template=
    ("You are given the following extracted parts of marketing documents and a question. "
    "Read the following documents carefully. \n"

    "\n=========\n"

    "\nRelevant Sources: \n"

    "{context}"

    "\n=========\n"

    "You are an expert customer support agent with excellent attention to detail. "
    "You should ONLY use the information in 'Relevant Sources' section provided above while answering. "
    "DON'T use your prior knowledge to answer customer question. "
    "Always provide a short conversational answer with maximum clarity. "
    "When it comes to product related query I always want you to double check you're answering with correct product name and details. \n"
    "In a follow up question, answer to the same product customer is talking about."
    "If you don't know the answer or if sufficient details are not present in the relevant documents, "
    "DON'T try to make up an answer, it will create legal complications for misdirecting customers, just redirect the customer to other agents available through phone "
    "at 1-800-431-7678. The agents are available from Monday to Friday 9 AM - 5 PM (ET). \n"

    "\n=========\n"

    "\nCurrent Chat: \n"
    "\n{chat_history}\n"),
    input_variables=['context','chat_history']
)

query_rewriter_template=PromptTemplate(
    template=(
    "Modify the latest Human question if it is a follow up question for better search retrieval. If it is a greeting or a new topic (not a follow up question) then rewrite it in a better way without using the conversation history.\n" 
    "For example: \n"

    "\n======\n"
    "Example 1 (For a Follow-up question in same topic) - If the follow-up question does not have the poduct name, include the product name as it helps for better retrieval: \n"
    "Human: Tell me about Honey bunches cereal?\n"
    "AI: Honey Bunches of Oats Almonds cereal is filled with crispy corn flakes, crunchy oat granola clusters, sliced almonds and topped with a touch of honey to help you feel energized.\n"
    "Human: what is the price?\n\n"

    "Final Query: What is the price of Honey bunches cereal?\n\n"

    "Example 2 (For a question in a new topic) - No need to include the previous product name as the question is about a new topic: \n"
    "Human: Tell me about standard Honey bunches cereal?\n"
    "AI: Honey Bunches of Oats Almonds cereal is filled with crispy corn flakes, crunchy oat granola clusters, sliced almonds and topped with a touch of honey to help you feel energized.\n"
    "Human: What size does weetabix come in?\n\n"

    "Final Query: What is the package size of weetabix?\n"

    "\n======\n"
    "Current Conversation: \n"
    "{memory}\n"
    "Final Query: "),
    input_variables=['memory']
)

hallucination_template = PromptTemplate(
    template=(
    "Evaluate if the last answer generated by AI is present inside the sources and if the last human query is related to a product, "
    "additionally, check if the answer is generated for the correct product. Answer 0 if the asnwer is generated from source AND "
    "for the correct product. Answer 1 if the answer is irrelevant to source or product. "
    "Do not leave it empty give your best judgement. \n"

    "\n======\n"
    "Sources: \n"
    "{sources}"
    "\n======\n"
    "Conversation History: \n"
    "{chat_history}"
    "\n======\n"
    "Answer: "),
    input_variables = ['sources','chat_history']
)

classification_template = PromptTemplate(
    template=(
    "You're a classification bot trained to redirect Post Consumer Brand (PCB)'s customer questions. "
    "Post Consumer Brands produce iconic breakfast cereals, snacks and pet food. "
    "The brands under PCB include: Alpen Muesli, Barbara's, Better Oats, Bran Flakes, Coco Wheats, Disney100, Farina Mills, Golden Crisp, Grape-Nuts, Great Grains, Honey Bunches of Oats, Honey Maid S’mores, Honeycomb, Malt-O-Meal Hot, Malt-O-Meal, Mom's Best, Honey Ohs!, Oreo O’s, Pebbles, Premier Protein, Puffins, Raisin Bran, Shredded Wheat, Snoop Cereal, Sweet Dreams, Sweet Home Farm, Uncle Sam, Waffle Crisp, Weetabix."
    "Classify the customer question into one of the following 4 classes and further into their "
    "subclasses if possible so that it can be answered by the correct department: \n\n"

    "* 'query': If the customer is greeting or asking general questions about Post Consumer Brands or "
    "about details like pricing, ingredients, usage, etc., of its products. \n"
    "Subclasses: greetings, product_query, company_related_query, and unknown. \n\n"

    "* 'complaint': If the customer has any issues with orders like quality concerns, "
    "shipment delays, etc., that need human attention. \n"
    "Subclasses: create_ticket, view_ticket, update_ticket, and unknown. \n\n"

    "* 'order': If the customer wants to perform order related operations like "
    "create, view, update or cancel orders. \n"
    "Subclasses: create_order, view_order, update_order, cancel_order, and unknown. \n\n"

    "* 'tangential': If the customer asks questions unrelated to post consumer brands or its products. \n"
    "Subclasses: unable_to_classify, and unrelated. \n\n"

    "Return your answer only in 'class label'-'sub_class label' format.\n"
    "Query: {query} \n"
    "Answer: "),
    input_variables=['query']
)


human_prompt_template = HumanMessagePromptTemplate.from_template(template = '{query}\nAI: ', input_variables = ['query'])
chat_template = ChatPromptTemplate.from_messages([system_prompt_template,human_prompt_template])
hallucination_check = False
model_name = os.getenv('GPT_3_MODEL')
salesforce_client = salesforce_connect()
vector_store = vector_search(database_name='knowledge',collection_name='products',index_name='searchindex')
chatlog_collection = mongodb_collection(database_name='chat',collection_name='chatlog')
flagged_response_collection = mongodb_collection(database_name='chat',collection_name='flagged_chat')
llm = ChatOpenAI(model = model_name, temperature = 0)
llm_chain = LLMChain(llm=llm, prompt=chat_template)

org_name = 'Post Consumer Brands'

about_org = ("Post Consumer Brands sells only the following iconic breakfast cereals, snacks and pet food. "
    "The brands under PCB include: Alpen Muesli, Barbara's, Better Oats, Bran Flakes, Coco Wheats, Disney100, Farina Mills, Golden Crisp, Grape-Nuts, Great Grains, Honey Bunches of Oats, Honey Maid S’mores, Honeycomb, Malt-O-Meal Hot, Malt-O-Meal, Mom's Best, Honey Ohs!, Oreo O’s, Pebbles, Premier Protein, Puffins, Raisin Bran, Shredded Wheat, Snoop Cereal, Sweet Dreams, Sweet Home Farm, Uncle Sam, Waffle Crisp, Weetabix.\n")


''' print("connection successful")
def get_query(prompt):
    query_chain = create_sql_query_chain(llm,db,k=10)
    assistant_response = {'SQL Query':query_chain.invoke({'question':prompt})}
    chatbot_response = assistant_response['SQL Query']
    return chatbot_response

def get_query_result(prompt):
    query = get_query(prompt)

    exec_sql = conn.cursor()

    exec_sql.execute(query)
    records = exec_sql.fetchall()
    exec_sql.close()
    return(records)

def interpret(prompt):
    full_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, use_query_checker=True, return_intermediate_steps=True)
    assistant_response = full_chain(prompt)
    return assistant_response['result'] '''


# Create your views here.
def login_user(request):
    if request.user.is_authenticated:
        return redirect('signed')
    
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            csrf_token = request.COOKIES.get('csrftoken')
            return redirect('signed')
        else:
            messages.success(request, ('There was an error, try again!'))
            return redirect('login')
    
    else:
        return render(request, 'login.html', {})
    
def home_page(request):
    return render(request, 'home.html', {})

def signup_user(request):
    return render(request, 'signup.html', {})

def signed(request):
    if request.method=="POST":
        logout(request)
        return redirect('login')
    else:
        if request.user.is_authenticated:
            is_superuser = request.user.is_superuser
            group_name = request.user.groups.filter(name='admins').exists()
            return render(request, 'signedin.html', {'is_superuser': is_superuser, 'group_name': group_name})
        else:
            return redirect('login')
        
def chatbot(request):
    if not request.user.is_authenticated:
        return redirect('login')
    else:
        if request.method=="POST":
            request.session.create()
            session_id = request.session.session_key
            username = request.user.username

            user_prompt= request.POST.get('prompt')
            assistant_response = chat_response(llm = llm,llm_chain = llm_chain,
                                            vector_store = vector_store, 
                                            session_id = 'test-session',
                                            query = user_prompt,
                                            about_org = about_org, org_name = org_name,
                                            query_rewriter_template = query_rewriter_template,
                                            hallucination_template = hallucination_template,
                                            classification_template = classification_template,
                                            chatlog_collection = chatlog_collection,
                                            hallucination_check = hallucination_check, 
                                            username = 'testuser')
            chatbot_response = assistant_response['text']
            response_data = {
                'user': user_prompt,
                'bot': chatbot_response
            }
            print(chatbot_response)
            # print(repr(chatbot_response))
            new_chatlog = chatlog(session_id=session_id, username=username, prompt=user_prompt, response=chatbot_response)
            new_chatlog.save()
            
            return JsonResponse(response_data)
    return render(request, 'chatbot.html')