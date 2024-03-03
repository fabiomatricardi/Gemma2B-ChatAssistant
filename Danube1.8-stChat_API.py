import streamlit as st
# Chat with an intelligent assistant in your terminal
from openai import OpenAI
from time import  sleep
import datetime

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#AVATARS  👷🐦
av_us = '👷'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = '🦖'

# Set the webpage title
st.set_page_config(
    page_title="Your LocalGPT with 🌍 h2o-danube-chat",
    page_icon="🦖",
    layout="wide")

# Create a header element
st.header("Your own LocalGPT with 🌍 h2o-danube-1.8b-chat")
st.markdown("#### :green[*h2o-danube-1.8b-chat Q4 GGUF - the best 1.8B model?*]")


# create THE SESSIoN STATES
if "logfilename" not in st.session_state:
## Logger file
    tstamp = datetime.datetime.now()
    tstamp = str(tstamp).replace(' ','_')
    tstamp = str(tstamp).replace(':','_')
    logfile = f'{tstamp[:-7]}_log.txt'
    st.session_state.logfilename = logfile
    sleep(2)
    #Write in the history the first 2 sessions
    writehistory(st.session_state.logfilename,f'Your own LocalGPT with 🌍 h2o-danube-1.8b-chat\n---\n🧠🫡: You are a helpful assistant.')    
    writehistory(st.session_state.logfilename,f'🦖: How may I help you today?')

if "len_context" not in st.session_state:
    st.session_state.len_context = 0

if "limiter" not in st.session_state:
    st.session_state.limiter = 0

if "bufstatus" not in st.session_state:
    st.session_state.bufstatus = "**:green[Good]**"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1

if "maxlength" not in st.session_state:
    st.session_state.maxlength = 350

# Point to the local server
# Change localhost with the IP ADDRESS of the computer acting as a server
# itmay be something like "http://192.168.1.52:8000/v1"
client = OpenAI(base_url="http://localhost:8000/v1", 
                api_key="not-needed")

# CREATE THE SIDEBAR
with st.sidebar:
    st.markdown("""### 🦖 h2o-danube-1.8b-chat
- ConversationBuffer+Limiter
- Real streaming output
- HyperParameters""", unsafe_allow_html=True)
    mytokens = st.markdown(f"""**Context turns** {st.session_state.len_context}  **n_ctx**: 16384""")
    st.session_state.temperature = st.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.1, step=0.02)
    st.session_state.limiter = st.slider('Turns:', min_value=7, max_value=17, value=12, step=1)
    st.session_state.maxlength = st.slider('Length reply:', min_value=150, max_value=500, 
                                           value=350, step=50)
    st.markdown(f"Buffer status: {st.session_state.bufstatus}")
    st.markdown(f"**Logfile**: {st.session_state.logfilename}")
    btnClear = st.button("Clear History",type="primary", use_container_width=True)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "assistant", "content": "How may I help you today?"}
    ]

def clearHistory():
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "assistant", "content": "How may I help you today?"}
    ]
    sleep(2)
    st.session_state.len_context = len(st.session_state.messages)
if btnClear:
      clearHistory()  
      sleep(2)
      st.session_state.len_context = len(st.session_state.messages)

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages[1:]:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here. Shift+Enter to add a new line", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user", avatar=av_us):
        st.markdown(user_prompt)
        writehistory(st.session_state.logfilename,f'👷: {user_prompt}')

    
    with st.chat_message("assistant",avatar=av_ass):
        with st.spinner("Thinking..."):
            response = ''
            conv_messages = []
            st.session_state.len_context = len(st.session_state.messages) 
            # Checking context window for the LLM, not for the chat history to be displayed
            if st.session_state.len_context > st.session_state.limiter:
                st.session_state.bufstatus = "**:red[Overflow]**"
                # this will keep 5 full turns into consideration 
                x=st.session_state.limiter-5
                conv_messages.append(st.session_state.messages[0])
                for i in range(0,x):
                    conv_messages.append(st.session_state.messages[-x+i])
                print(len(conv_messages))
                completion = client.chat.completions.create(
                    model="local-model", # this field is currently unused
                    messages=conv_messages,
                    temperature=st.session_state.temperature,
                    #repeat_penalty=1.4,
                    stop=['<|im_end|>','</s>',"<end_of_turn>"],
                    max_tokens=st.session_state.maxlength,
                    stream=True,
                )
                response = st.write_stream(completion)
                writehistory(st.session_state.logfilename,f'🦖: {response}') 
            else:
                st.session_state.bufstatus = "**:green[Good]**"
                completion = client.chat.completions.create(
                    model="local-model", # this field is currently unused
                    messages=st.session_state.messages,
                    temperature=st.session_state.temperature,
                    #repeat_penalty=1.4,
                    stop=['<|im_end|>','</s>',"<end_of_turn>"],
                    max_tokens=st.session_state.maxlength,
                    stream=True,
                )
                response = st.write_stream(completion)
                writehistory(st.session_state.logfilename,f'🦖: {response}') 
            
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    st.session_state.len_context = len(st.session_state.messages)
