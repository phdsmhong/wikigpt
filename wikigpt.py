from  PIL import Image
import requests
import wikipedia
from bs4 import BeautifulSoup
import os
import time
import pickle
import streamlit as st
from datetime import datetime
from streamlit_chat import message

from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from wiki_content import get_wiki

# Name in the sidebar
st.set_page_config(page_title = 'Custom GPT')
###################
def sidebar_bg():
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url("https://cdn.pixabay.com/photo/2013/09/26/00/21/swan-186551_1280.jpg")
               }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar_bg()
###############################################
###############################################
# NAVIGATION BAR 
#https://discuss.streamlit.io/t/the-navigation-bar-im-trying-to-add-to-my-streamlit-app-is-blurred/24104/3
# First clean up the bar 
st.markdown(
"""
<style>
header[data-testid="stHeader"] {
    background: none;
}
</style>
""",
    unsafe_allow_html=True,
)
# Then put the followings (Data Prof: https://www.youtube.com/watch?v=hoPvOIJvrb8)
# Background color에 붉은빛이 약간 들어간 #ffeded
# https://color-hex.org/color/ffeded
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FFF7F7;">
  <a class="navbar-brand" href="https://digitalgovlab.com" target="_blank">Digital Governance Lab</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
    </ul>
  </div>
</nav>
<style>
    .navbar-brand{
    color: #89949F !important;
     }
    .nav-link disabled{
    color: #89949F !important;
     }
    .nav-link{
    color: #89949F !important;
     }
</style>
""", unsafe_allow_html=True)
#############################################################
#############################################################
##############################################################
#--- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
#########################################################################
# LOGO
#https://pmbaumgartner.github.io/streamlitopedia/sizing-and-images.html
image = Image.open('logo_digitalgovlab_v2.jpg')
st.image(image, caption='')
########################
st.markdown(""" <style> .font2 {
     font-size:30px ; font-family: 'Cooper Black'; color: #000000;} 
     </style> """, unsafe_allow_html=True)
st.markdown('<p class="font2">Wiki GPT에 오신 것을 환영합니다</p>', unsafe_allow_html=True) 
st.markdown(""" <style> .font3 {
     font-size:20px ; font-family: 'Cooper Black'; color: #000000;} 
     </style> """, unsafe_allow_html=True)
#st.markdown("##")
#st.markdown('<p class="font3">Wiki GPT 소개</p>', unsafe_allow_html=True) 
url = "https://platform.openai.com/account/api-keys"
st.markdown("""
Wiki GPT는 기존 챗GPT에 위키백과의 내용을 in-context/embedding 방식으로 추가 학습시킨 AI입니다.  \
23년 4월 현재 챗GPT에 비해 훨씬 더 전문적인 질문에 대해서 답변을 할 수 있습니다. \
랩의 예산 사정상 무료 이용기간은 종료되었습니다. \
직접 OpenAI의 API를 생성하여 사용하시기 바랍니다. [API Key 생성하러 가기](%s)
""" % url)
################################
#https://medium.com/@shashankvats/building-a-wikipedia-search-engine-with-langchain-and-streamlit-d63cb11181d0

global embeddings_flag
embeddings_flag = False

######################################
####################################################
#buff, col, buff2 = st.columns([1,3,1])
#st.markdown("##")
st.markdown("---")
st.markdown("OpenAI API Key를 입력해주세요 (API Key는 sk-로 시작합니다)")
openai_key = st.text_input(label=" ", label_visibility="collapsed")
os.environ["OPENAI_API_KEY"] = openai_key

if len(openai_key):
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    if 'all_messages' not in st.session_state:
        st.session_state.all_messages = []

    def build_index(wiki_content):
        print("building index .....")
        text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,  
            )  
        texts = text_splitter.split_text(wiki_content)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        with open("./embeddings.pkl", 'wb') as f:
            pickle.dump(docsearch, f)
        return embeddings, docsearch

    # Create a function to get bot response
    def get_bot_response(user_query, faiss_index):
        docs = faiss_index.similarity_search(user_query, K = 6)
        main_content = user_query + "\n\n"
        for doc in docs:
            main_content += doc.page_content + "\n\n"
        messages.append(HumanMessage(content=main_content))
        ai_response = chat(messages).content
        messages.pop()
        messages.append(HumanMessage(content=user_query))
        messages.append(AIMessage(content=ai_response))
        return ai_response

    # Create a function to display messages
    def display_messages(all_messages):
        for msg in all_messages:
            if msg['user'] == 'user':
                message(f"You ({msg['time']}): {msg['text']}", is_user=True, key=int(time.time_ns()))
            else:
                message(f"AI ({msg['time']}): {msg['text']}", key=int(time.time_ns()))

    # Create a function to send messages
    def send_message(user_query, faiss_index, all_messages):
        if user_query:
            all_messages.append({'user': 'user', 'time': datetime.now().strftime("%H:%M"), 'text': user_query})
            bot_response = get_bot_response(user_query, faiss_index)
            all_messages.append({'user': 'bot', 'time': datetime.now().strftime("%H:%M"), 'text': bot_response})

            st.session_state.all_messages = all_messages
            
    # Create a list to store messages
    messages = [
            SystemMessage(content="You are a Q&A bot and you will answer all the questions that the user has. If you dont know the answer, output '죄송합니다. 잘 모르겠습니다' .")
        ]
#############################################
    #st.markdown("##")
    st.markdown("---")
    col1, col2, col3 = st.columns([6, 0.1, 4.2])
    with col1: 
        st.markdown("관심 있으신 키워드를 입력해주세요")
        search = st.text_input(label="XX", label_visibility="collapsed", key= "1")
    with col3:
        st.markdown("몇 문장으로 요약할까요?")
        numsen = st.slider('XXXX', min_value=1, max_value=8, value=3, step=1, label_visibility="collapsed", key = '33')
   
    if len(search):
        wiki_content, summary = get_wiki(search, numsen)

        if len(wiki_content):
            st.write(summary)

            
