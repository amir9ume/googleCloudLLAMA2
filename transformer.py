
'''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

'''


access_token='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'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_id = "tiiuae/falcon-7b-instruct"

# Define the path to save the model and tokenizer
model_path = "./models/hf-frompretrained-download/Llama-2-7b-chat-hf"

# Function to check if the model files exist
def model_files_exist(model_path):
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    return True

# Check if the model and tokenizer already exist
if not model_files_exist(model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("Downloading and saving model and tokenizer...")
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token= access_token)

    # Save your files to the specified directory
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
else:
    print("Model and tokenizer already exist. Loading from saved files...")

# Reload your files offline
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


#tokenizer = AutoTokenizer.from_pretrained(model_id, token= access_token)
#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    torch_dtype=torch.bfloat16,trust_remote_code= True,low_cpu_mem_usage = True,
#    token= access_token
#).cpu()

#torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage = True).cpu()

#from accelerate import disk_offload
#disk_offload(model=model, offload_dir="offload")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    messages = data.get('messages', [])
    
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

