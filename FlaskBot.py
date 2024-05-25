from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


modelpath="/media/parmpal/Data/Models/Torch/LLAMA/llama-2-7b-chat.Q4_0.gguf"
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    # model_path="drive/MyDrive/llama-2-7b-chat.Q4_0.gguf", # downloaded from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
    model_path=modelpath, # downloaded from https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
    temperature=0.5,
    max_tokens=2000,
    top_p=1,
    n_ctx=4000,
    # callback_manager=callback_manager,
    # verbose=True, # Verbose is required to pass to the callback manager
)
prompt=""

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def chat():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(message):
    print('Received message: ' + message)
    response = get_chatbot_response(message)  # Replace with your response method
    emit('response', response)

def get_chatbot_response(message):
    # Your method to generate a response from the chatbot
    # Replace this with the actual implementation deployed at the server
    global prompt
    prompt += "\n [INST] " + message + " [/INST]\n"
    response=llm(prompt)
    prompt += "\n" + response + "\n"
    return response

if __name__ == '__main__':
    socketio.run(app, debug=True,allow_unsafe_werkzeug=True)