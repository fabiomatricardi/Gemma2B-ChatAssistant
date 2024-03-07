# Gemma2B-ChatAssistant
Maybe Gemma-2B-IT is the smallest and best chat friend you can have.

Or maybe one the listed above Small Language Models:
- Cosmo-1b
- Gemma-2b
- Danube-1.8b-chat
- TinyLlama-chat-1.1b

Create a new folder for this project, create a virtual environment, and activate it (the instructions here are for Windows users).
I am still using Python 3.10, but also 3.11 will be ok. Do not use 3.12 if you intend to use Sentencepiece (it is not yet supported).
```
mkdir HAT
cd HAT
python -m venv venv

venv\Scripts\activate
```


Now that the Virtual Environment is activated, you need to install the only 3 libraries required:
```
pip install llama-cpp-python[server]==0.2.53
pip install openai
pip install streamlit
```

Download the models and save them into the `model` subfolder.
- https://huggingface.co/brittlewis12/h2o-danube-1.8b-chat-GGUF
- https://huggingface.co/asedmammad/gemma-2b-it-GGUF/tree/main
- https://huggingface.co/tsunemoto/cosmo-1b-GGUF/tree/main

I am giving you here the link to 3 models (danube-1.8B-chat Q5_K_M, Gemma-2B-it Q4_K_M, and Cosmo-1b Q4_K_M): but remember that you will be able to use much more than them
Now we are all set.
The model files of cosmo-1b.Q4_K_M.gguf, gemma-2b-it.q4_K_M.gguf and h2o-danube-1.8b-chat.Q5_K_M.gguf must be in a subfolder called `model`.



To run the model as an API and then to run streamlit we need two terminal windows: one will start the FastAPI server, and the other one will act as a Streamlit GUI server. Both terminals must have the venv activated.

On the terminal on the left run the llama-cpp-server with this command:
```
python -m llama_cpp.server --host 0.0.0.0 --model model/h2o-danube-1.8b-chat.Q5_K_M.gguf --n_ctx 16384
```

On the terminal to the right start the Streamlit server with this command:
```
streamlit run .\Danube1.8-stChat_API.py
```


