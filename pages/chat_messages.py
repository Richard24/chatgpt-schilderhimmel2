import streamlit as st
import pymongo
import mysql.connector
from mysql.connector import Error
import pandas as pd
import sshtunnel
import pymysql.cursors


# Set SSH and Tunnel timeout
sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0

# Use try-except to catch and display errors in the Streamlit app
try:
    # Establish an SSH tunnel
    with sshtunnel.SSHTunnelForwarder(
        ('ssh.pythonanywhere.com'),
        ssh_username='schwabkunststoff1', ssh_password=st.secrets["ssh_password"],
        remote_bind_address=('schwabkunststoff1.mysql.pythonanywhere-services.com', 3306)
    ) as tunnel:
        st.write("SSH Tunnel established")
        
        connection = pymysql.connect(
            user='schwabkunststoff',
            passwd=st.secrets["ssh_password"],
            host='127.0.0.1', port=tunnel.local_bind_port,
            db='schwabkunststoff$chatgpt-schilderhimmel',
        )
        
        db = connection.cursor()
        db.execute("SELECT * FROM `chat-messages`")
        rows = db.fetchall()
        df = pd.DataFrame(rows, columns=[i[0] for i in db.description])
        st.dataframe(df, hide_index=True)
        
except Error as e:
    st.write("Error while connecting to MySQL:", e)
except Exception as e:
    st.write("An unexpected error occurred:", e)


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

# st.title("Chat Messages")

# Display the dataframe
# st.dataframe(df, hide_index=True)
