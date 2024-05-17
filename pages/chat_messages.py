import streamlit as st
import pymongo
import mysql.connector
from mysql.connector import Error
import pandas as pd
import sshtunnel

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

        # Connect to the MySQL database
        connection = mysql.connector.connect(
            user='schwabkunststoff',
            password=st.secrets["ssh_password"],
            host='127.0.0.1', 
            port=tunnel.local_bind_port,
            database='schwabkunststoff$chatgpt-schilderhimmel',
        )

        # Check if connected
        if connection.is_connected():
            st.write("Connected to MySQL database")
            
            # Create a cursor object using the connection
            cursor = connection.cursor()
            
            # Execute the SQL query
            cursor.execute("SELECT * FROM `chat-messages` ORDER BY `datetime` DESC")
            
            # Fetch all rows from the executed query
            rows = cursor.fetchall()
            
            # Fetch column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert the rows to a pandas DataFrame
            df = pd.DataFrame(rows, columns=column_names)
            
            # Display the DataFrame in the Streamlit app
            st.write(df)
            
            # Close the cursor
            cursor.close()
            
        else:
            st.write("Failed to connect to MySQL database")
        
        # Close the connection
        connection.close()
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

st.title("Chat Messages")

# Display the dataframe
st.dataframe(df, hide_index=True)
