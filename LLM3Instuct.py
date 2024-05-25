import os
import time

import transformers
import torch

def loadModel():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    st=time.time()
    device_map={'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map=device_map, #"auto",
        # model_kwargs={"torch_dtype": torch.bfloat16},
        model_kwargs={"torch_dtype": torch.bfloat16,"offload_folder":"offload"}
    )
    print("Loading time",time.time()-st)
    return pipeline

pipeline=loadModel()
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
def Inference(pipeline):
    #system user assistant
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    print("Infernce")
    st=time.time()
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )


    print("Generation time time",time.time()-st)
    print(outputs)
    print(outputs[0].keys())
    print(len(outputs))
    print(outputs[0]["generated_text"][len(prompt):])


def chat(pipeline):
    # system user assistant
    messages = [
        {"role": "system",
         "content": "You are a Hospital assistant you will recive message and ask how can i help you. "
                    "the patient will respond with appointmetn requirement. you will ask for the department"},
        {"role": "user", "content": "hello"},
    ]
    messages = [
        {"role": "system",
         "content": """You are a ABC Hospital assistant you will start the chat and Welcome message and ask how can i help you. 
         and deal the patient as you can. you will need to ask for the department don't give them the list of department to choose based on symptoms.
         we have Surgeons, Psychiatrists, Cardiologists departments
         for Surgeons we have dr John available from 9am to 12pm then 3pm to 6pm
         for Psychiatrists we have dr Win available from 9am to 1pm
         for Cardiologists we have dr John available from 9am to 2pm 
         Also Ask for the best time suits to patient after that ask for name, dob and phone number one by one.
         Once we got all Thanks the paitent and let them know the appointment is booked. and end the conversation
         Make the response simple and 1 or 2 lines only. don't be in hurry. try to make simple for user
         Don't ask the User What he should ask"""},

    ]

    while True:
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("Infernce")
        st = time.time()
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print("Generation time time", time.time() - st)
        print(outputs)
        # print(outputs[0].keys())
        # print(len(outputs))
        # print(outputs[0]["generated_text"][len(prompt):])
        response=outputs[0]["generated_text"][len(prompt):]
        messages.append({"role": "assistant","content":response})
        print(response)
        for mesagge in messages:
            # print(mesagge)
            # print(type(mesagge))
            print(f"{mesagge['role']}:{mesagge['content']}")
        messages.append({"role": "user","content":input()})

# Inference(pipeline)
chat(pipeline)