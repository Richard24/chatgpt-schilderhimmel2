import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import uuid
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter


pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pc.Index("chatgpt-schilderhimmel")

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

client = OpenAI()


def load_text(text):
    loader = TextLoader(text)
    docs = text.load_and_split()
    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(docs)
    return splits


def embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def vector_store(vector, metadata):
    index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": metadata
            }
        ]
    )


def search_vector(vector):
    result = index.query(
        vector=vector,
        top_k=3,
        include_metadata=True

    )
    return result


def filter_vector(vector, source):
    results = index.query(
        vector=vector,
        filter={
            "source": source
        },
        top_k=1000
        # include_metadata=True
    )
    return results


def delete_vector(ids):
    index.delete(ids=[ids])
    print("delete success")

st.title("Schilderhimmel FAQ")

# upload text
with st.form("my_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload Text", type="txt")
    submitted = st.form_submit_button("UPLOAD")
    if submitted and uploaded_file is not None:
        file_name = uploaded_file.name
        documents = uploaded_file.read().decode(errors='ignore')

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,  # Adjust based on your token limits
        #     chunk_overlap=100,  # Helps maintain context across chunks
        #     separators=["\n\n", "\n", " "],  # Prioritize structured breaks
        # )
        
        # data_splits = text_splitter.create_documents([documents])
        
        text_splitter = CharacterTextSplitter(
            separator="Question:",  # Split at each "Question:"
            chunk_size=1000,  # Adjust based on your model's token limit
            chunk_overlap=0,  # No overlap needed since each Q&A is independent
        )

        chunks = text_splitter.split_text(documents)

        
        formatted_chunks = [("Question:" + chunk).strip() for chunk in chunks if chunk.strip()]

        # st.write(formatted_chunks)

        text = " "
        text_embed = embedding(text)

        result = filter_vector(text_embed, file_name)
        ids = [item.id for item in result['matches']]

        if ids:
            delete_vector(ids)

        # store vector store
        for split in formatted_chunks:
            text = split

            metadata = {
                'source': file_name,
                'text': text
            }

            vector = embedding(text)

            vector_store(vector, metadata)

        st.toast('Store done: ' + file_name)

st.markdown("---")

search = st.text_input("Ask", )

if search:
    question_embedding = embedding(search)

    matches = search_vector(question_embedding)
    
    st.write(matches)

    text_context = matches['matches'][0]['metadata']['text'] + matches['matches'][1]['metadata']['text'] + matches['matches'][2]['metadata']['text']

    # Build prompt
    template = """Sie sind ein hilfreicher Assistent.
    Benutzen Sie die folgenden Kontextinformationen, um die Frage am Ende zu beantworten.
    Wenn Sie die Antwort nicht kennen, sagen Sie einfach: 'Leider kann ich Ihre Frage derzeit nicht beantworten. Bitte teilen Sie uns noch Ihre E-Mail Adresse mit, damit wir gleich Kontakt mit Ihnen aufnehmen können. Alternativ können Sie uns anrufen: 09604-5309873.' und versuchen Sie nicht, eine Antwort zu erfinden.
    Antwort auf Deutsch. Antworten Sie mit einer URL, falls im Kontext.
    Geben Sie im Chat niemals persönliche Antworten.
    HTML-Tag einschließen, falls im Kontext vorhanden.
    Wenn die Frage ein '@' enthält, antworte mit "Vielen Dank wir werden Sie schnellstmöglich per Mail kontaktieren. Bitte teilen Sie uns dazu Ihre Mail Adresse mit.".

    Kontext: {context}
    Frage: {question}
    Hilfreiche Antwort:"""

    prompt_template = PromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model='gpt-4')

    prompt = prompt_template.format(
        context=text_context,
        question=search
    )

    response = llm.predict(
        text=prompt
    )
    
    st.info(response)