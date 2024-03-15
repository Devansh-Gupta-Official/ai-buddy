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
import yagmail
from deep_translator import GoogleTranslator
from streamlit_feedback import streamlit_feedback
from lingua import Language, LanguageDetectorBuilder

st.set_page_config(
    "Multilingual Chat Bot 🤖", layout="wide", initial_sidebar_state="expanded")

load_dotenv()


cohere_api_key = os.getenv('cohere_api_key')
authorization = os.getenv('authorization')
# model= 'embed-multilingual-v3.0'
model_url=os.getenv('model_url')
auth=os.getenv('auth')

@st.cache_data
def bot():
    with open('task-1.pdf', "rb") as f:
        loader = PyPDFLoader(f.name)
        pages = loader.load_and_split()
        return pages


def send_email(emaill,report,auth):
    email = yagmail.SMTP(user="ojasfarm31@gmail.com", password=auth)
    email.send(to= emaill,
               subject="Your Conversations with Chat Bot",
               contents=f"Hi,  \n Check out your conversations with Chat Bot!! \nDo not reply back to this email. \n{report}\nRegards,\n<b>Chat Bot<b>")
    return True

col1,col2=st.columns([1,8])
with col2:
    st.header("Multilingual AI Chat Bot 🤖",divider="rainbow")
st.markdown(""" <style>
        section[data-testid="stSidebar"] {
        width: 500px !important;
        }
        </style>
        """, unsafe_allow_html=True,)
with st.sidebar:
    st.markdown(
        "<h1 style='font-style: italic; color: #F55F0E; text-align:center'>Welcome !<pre>  सवागतम !</h1>",
        unsafe_allow_html=True,)
    st.write("")

    st.subheader("ABOUT:")
    st.markdown(
        "Introducing the <strong>Multi-Lingual Chat Bot</strong> - Your gateway to questions in over <strong>100 languages.</strong>"
        "<br><br>Accessible <strong>24/7</strong>, it offers instant information, transcending language barriers,any question,anytime, where information knows no borders.",
        unsafe_allow_html=True,)

    st.write("")
    st.write("")
    email_input = st.text_input("Enter your Email ID:", placeholder="john@gmail.com")
    st.subheader("HOW TO USE: ")
    st.markdown(
        "<p style = 'cursor: default;'>1.Enter your Email ID."
        "<br>2. Type your question in the text box in your preferred language."
        "<br>3. Click 'Ask'."
        "<br>4.  Hurray! your answer's here!!",
        unsafe_allow_html=True,)

pages = bot()

initial = st.container()
message = f"""<div style='display:flex;align-items:center;margin-bottom:10px;'>
                    <img src='https://i.imgur.com/rKTnxVN.png' style='width:50px;height:50px;border-radius:50%;margin-right:10px;'>
                    <div style='background-color:st.get_option("theme.backgroundColor");border: 1px solid {st.get_option("theme.secondaryBackgroundColor")};border-radius:10px;padding:10px;'>
                    <p style='margin:0;font-weight:bold;'>Multilingual Chat Bot</p>
                    <p style='margin:0;color={st.get_option("theme.textColor")}'>Hi, How may I assist you today?</p>
                    </div>
                    </div>
          """
initial.write(message, unsafe_allow_html=True)

try:
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


    if(len(email_input)>11):
        messages_container = st.container()
        question = st.chat_input(
        placeholder="Type your query here...",
        key="input",
        max_chars=300 )


        if question:
            prompt.append({"role": "user", "content": question})
            chain_type_kwargs = {"prompt": PROMPT}
            with messages_container:
                user_message(question)
                botmsg = bot_message("...", bot_name="Chat Bot")
    

            qa = RetrievalQA.from_chain_type(
                llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
                chain_type="stuff",
                retriever=store.as_retriever(),
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True,
            )

            answer = qa({"query": question})
            result = answer["result"].replace("\n", "").replace("Answer:", "").replace("mentioned in the text:","").replace("According to the text you provided,","").replace("According to the provided text, ","")
            # payload = { "texts": [ f"{question}"] }
            # headers = {
            #     "accept": "application/json",
            #     "content-type": "application/json",
            #     "authorization": authorization
            # }
            # url = model_url
            # response = requests.post(url, json=payload, headers=headers)
            # parsed_data = simplejson.loads(response.text)
            # st.write(parsed_data)
            # language_code = parsed_data["results"][0]["language_code"]
            languages = [
    Language.AFRIKAANS,
    Language.ALBANIAN,
    Language.ARABIC,
    Language.ARMENIAN,
    Language.AZERBAIJANI,
    Language.BASQUE,
    Language.BELARUSIAN,
    Language.BENGALI,
    Language.BOSNIAN,
    Language.BULGARIAN,
    Language.CATALAN,
    Language.CHINESE,
    Language.CROATIAN,
    Language.CZECH,
    Language.DANISH,
    Language.DUTCH,
    Language.ENGLISH,
    Language.ESPERANTO,
    Language.ESTONIAN,
    Language.FINNISH,
    Language.FRENCH,
    Language.GANDA,
    Language.GEORGIAN,
    Language.GERMAN,
    Language.GREEK,
    Language.GUJARATI,
    Language.HEBREW,
    Language.HINDI,
    Language.HUNGARIAN,
    Language.ICELANDIC,
    Language.INDONESIAN,
    Language.IRISH,
    Language.ITALIAN,
    Language.JAPANESE,
    Language.KAZAKH,
    Language.KOREAN,
    Language.LATIN,
    Language.LATVIAN,
    Language.LITHUANIAN,
    Language.MACEDONIAN,
    Language.MALAY,
    Language.MAORI,
    Language.MARATHI,
    Language.MONGOLIAN,
    Language.PERSIAN,
    Language.POLISH,
    Language.PORTUGUESE,
    Language.PUNJABI,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SERBIAN,
    Language.SHONA,
    Language.SLOVAK,
    Language.SLOVENE,
    Language.SOMALI,
    Language.SOTHO,
    Language.SPANISH,
    Language.SWAHILI,
    Language.SWEDISH,
    Language.TAGALOG,
    Language.TAMIL,
    Language.TELUGU,
    Language.THAI,
    Language.TSONGA,
    Language.TSWANA,
    Language.TURKISH,
    Language.UKRAINIAN,
    Language.URDU,
    Language.VIETNAMESE,
    Language.WELSH,
    Language.XHOSA,
    Language.YORUBA,
    Language.ZULU
]

            detector = LanguageDetectorBuilder.from_languages(*languages).build()
            language = detector.detect_language_of(result)
            language_code=language.iso_code_639_1.name
            translated_text = GoogleTranslator(source="en", target=f"{language_code.lower()}").translate(f"{result}")
            with st.spinner("Loading response .."):
                input = "NULL"
                botmsg.update(translated_text)
            prompt.append({"role": "assistant", "content": translated_text})
            st.session_state["prompt"] = prompt

        if(len(prompt)>1):
            col1,col2,col3=st.columns([1,6,1])
            res=""
            with col3:
                email=st.button("Email Chat",help="Email conversations history to your email id")
                if(email):
                    ques=[]
                    for i in prompt:
                        if i.get("role") == "user":
                           ques.append("Query: ")
                           ques.append(i.get("content"))
                        if i.get("role") == "assistant":
                           ques.append("Answer: ")
                           ques.append(i.get("content").capitalize())
                    content = ""
                    for item in ques:
                            content += "<p>" + item + "</p>"
                    content+= "If you are not satisfied with the answers, kindly reply back to this email and let us know what you are looking for."
                    email_content = """
                                    <html>
                                      <body>
                                        <div style="position: relative; width: 800px; height: auto; background: url('https://img.freepik.com/premium-vector/blue-wave-abstract-design-soft-background_41084-392.jpg') repeat-y center center; background-size: 100% 100%; opacity: 0.7;">
                                          <div style="position: relative; color: black; font-size: 15px; font-weight: bold; text-align: left; padding: 20px; box-sizing: border-box;">
                                            """ + content + """
                                          </div>
                                        </div>
                                      </body>
                                    </html>
                                    """

                    res=send_email(email_input,email_content,auth)
            with col2:
                if(res):
                    st.success("Email sent successfully !",icon="✅")
                    feed=streamlit_feedback(
                    feedback_type="faces",
                    optional_text_label="[Optional] Please provide an explanation",
                    key="feedback")
                    feed
    else:
        initia = st.container()
        messag = f"""<div style='display:flex;align-items:center;margin-bottom:10px;'>
                        <img src='https://i.imgur.com/rKTnxVN.png' style='width:50px;height:50px;border-radius:50%;margin-right:10px;'>
                        <div style='background-color:st.get_option("theme.backgroundColor");border: 1px solid {st.get_option("theme.secondaryBackgroundColor")};border-radius:10px;padding:10px;'>
                        <p style='margin:0;font-weight:bold;'>Chat Bot</p>
                        <p style='margin:0;color={st.get_option("theme.textColor")}'>Enter  your email in left sidebar to ask you queries.</p>
                        </div>
                        </div>
              """
        initia.write(messag, unsafe_allow_html=True)


# except:
#     st.error("API KEY EXHAUSTED!",icon="🚨")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
