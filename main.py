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

pc = Pinecone(api_key="950b50af-bfe0-48d5-bd99-d3e0ea6c1eb2")
index = pc.Index("chatgpt-schilderhimmel")

os.environ["OPENAI_API_KEY"] = "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"

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
        top_k=1,
        include_metadata=True

    )
    return result


def filter_vector(vector, source):
    results = index.query(
        vector=vector,
        filter={
            "source": source
        },
        top_k=100,
        include_metadata=True
    )
    return results


def delete_vector(file_name):
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

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        data_splits = text_splitter.create_documents([documents])

        # st.write(data_splits)

        text = " "
        text_embed = embedding(text)

        result = filter_vector(text_embed, file_name)
        ids = [item.id for item in result['matches']]

        if ids:
            delete_vector(file_name)

        # store vector store
        for split in data_splits:
            text = split.page_content

            metadata = {
                'source': file_name,
                'text': text
            }

            vector = embedding(split.page_content)

            vector_store(vector, metadata)

        st.toast('Store done: ' + file_name)

st.markdown("---")

search = st.text_input("Ask", )

if search:
    question_embedding = embedding(search)

    matches = search_vector(question_embedding)
    
    # print(matches)

    text_context = matches['matches'][0]['metadata']['text']

    # Build prompt
    template = """Sie sind ein hilfreicher Assistent. Benutzen Sie die folgenden Kontextinformationen, um die Frage am Ende zu beantworten. Wenn Sie die Antwort nicht kennen, sagen Sie einfach, dass Sie es nicht wissen, und versuchen Sie nicht, eine Antwort zu erfinden. Antwort auf Deutsch. Antworten Sie mit einer URL, wenn im Kontext. Geben Sie im Chat niemals persönliche Antworten.
    {context}
    Frage: {question}
    Hilfreiche Antwort:"""

    prompt_template = PromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)

    prompt = prompt_template.format(
        context=text_context,
        question=search
    )

    response = llm.predict(
        text=prompt
    )
    st.info(response)
