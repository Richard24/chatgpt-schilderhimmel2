import streamlit as st
import pymongo
import mysql.connector
from mysql.connector import Error
import pandas as pd
import sshtunnel

sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0

try:
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username='schwabkunststoff1', ssh_password=st.secrets["ssh_password"],
        remote_bind_address=('schwabkunststoff1.mysql.pythonanywhere-services.com', 3306)
    ) as tunnel:
        connection = mysql.connector.connect(
            user='schwabkunststoff',
            password=st.secrets["ssh_password"],
            host='127.0.0.1', 
            port=tunnel.local_bind_port,
            database='schwabkunststoff$chatgpt-schilderhimmel',
        )
        
        # Check if connected
        if connection.is_connected():
            print("Connected to MySQL database")
            
            # Create a cursor object using the connection
            cursor = connection.cursor()
            
            # Execute the SQL query
            cursor.execute("SELECT * FROM `chat-messages` ORDER BY `datetime` DESC")
            
            # Fetch all rows from the executed query
            rows = cursor.fetchall()
            
            # Iterate through the rows and print them
            # for row in rows:
            #     print(row)
            
            # Fetch column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert the rows to a pandas DataFrame
            df = pd.DataFrame(rows, columns=column_names)
            
            # Close the cursor
            cursor.close()
            
        # Do stuff
        connection.close()
except Error as e:
    print("Error while connecting to MySQL", e)

# Replace these values with your MongoDB connection string and database name
# mongo_uri = st.secrets["mongo_uri"]
# database_name = "chatgpt-schilderhimmel"

# Connect to MongoDB
# client = pymongo.MongoClient(mongo_uri)
# database = client[database_name]

# Specify the collection
# collection_name = "chat-messages"
# collection = database[collection_name]

# Retrieve all documents from the collection
# all_messages = collection.find().sort("datetime", pymongo.DESCENDING)

st.title("Chat Messages")

# Display the dataframe
st.dataframe(df, hide_index=True)
