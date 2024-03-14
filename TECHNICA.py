import streamlit as st
from langchain_community.embeddings.cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from db_chat import user_message, bot_message
from langchain_community.document_loaders import PyPDFLoader
import requests
import simplejson
from dotenv import load_dotenv
import os

load_dotenv()


cohere_api_key = os.getenv('cohere_api_key')
authorization = os.getenv('authorization')
# model= 'embed-multilingual-v3.0'
model_url=os.getenv('model_url')

@st.cache_data
def bot():
    with open('samsung.pdf', "rb") as f:
        loader = PyPDFLoader(f.name)
        pages = loader.load_and_split()
        return pages
pages = bot()


try:
    email_input = "hi"
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )

    prompt_template = """Text: {context}
    Question: {question}
    You are replying as an AI Chat Bot n,
    answer the question without using any vulgularity."""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    prompt = st.session_state.get("prompt", None)
    if prompt is None:
        prompt = [{"role": "system", "content": prompt_template}]
    for message in prompt:
        if message["role"] == "user":
            user_message(message["content"])
        elif message["role"] == "assistant":
            bot_message(message["content"], bot_name="Chat Bot")


    if(len(email_input)==2):

        question = st.text_input(label='chat',
        placeholder="Type your query here...",
        key="input",
        max_chars=300 )

        if question:
            prompt.append({"role": "user", "content": question})
            chain_type_kwargs = {"prompt": PROMPT}
    

            qa = RetrievalQA.from_chain_type(
                llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
                chain_type="stuff",
                retriever=store.as_retriever(),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True,
            )

            answer = qa({"query": question})
            result = answer["result"].replace("\n", "").replace("Answer:", "").replace("mentioned in the text:","").replace("According to the text you provided,","").replace("According to the provided text, ","")
            payload = { "texts": [ f"{question}"] }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": authorization
            }
            url = model_url
            response = requests.post(url, json=payload, headers=headers)
            parsed_data = simplejson.loads(response.text)
    
            prompt.append({"role": "assistant", "content": result})
            st.session_state["prompt"] = prompt

            st.write(result)
        
    else:
        st.write("done")

# except:
#     st.error("API KEY EXHAUSTED!",icon="🚨")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
