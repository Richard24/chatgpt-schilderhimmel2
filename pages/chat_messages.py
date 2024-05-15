import streamlit as st
import pymongo

# Replace these values with your MongoDB connection string and database name
mongo_uri = st.secrets["mongo_uri"]
database_name = "chatgpt-schilderhimmel"

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)
database = client[database_name]

# Specify the collection
collection_name = "chat-messages"
collection = database[collection_name]

# Retrieve all documents from the collection
all_messages = collection.find()

st.title("Chat Messages")

# Display the dataframe
st.dataframe(all_messages)
