import argparse
import os
import json
import transformers
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import utils
import loraft
#import datautils  # Ensure utils.py has required functions

# FIXME: add API keys 
from GET_API import get_api_tokens

get_api_tokens()

# FIXME: remove debugger
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['run_ft', 'run_icl', 'chat', 'plot', 'cache'])
parser.add_argument('--model', choices=['bert-tiny', 'bert-medium'], default='bert-tiny')
parser.add_argument('--dataset', default = '../fitness.json')
parser.add_argument('--k', default='0', type=int)
parser.add_argument('--loramode', choices=['lora2', 'lora4', 'lora8', 'lora16'], default = 'lora4')
parser.add_argument('--prompt', default='qa')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--output', default='plot.png')
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

os.environ["DEVICE"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent warnings

######################################################################
# FIXME: move langchain to external file
# FIXME: move langchain to other file
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import LLMChain, RetrievalQA

from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory 

from langchain.indexes import VectorstoreIndexCreator

def langchainer():
    # Load JSON doc 
    loader = JSONLoader(
        file_path = "../fitnessdoc.json",
        jq_schema = ".[].content"
    )

    # Load data 
    data = loader.load()

    # Chunk data 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)

    #return Chroma.from_documents(documents = all_splits, embedding = GPT4AllEmbeddings)

    index = VectorstoreIndexCreator().from_loaders([loader])
    return index 

######################################################################

# Define functions for training and in-context learning

def fine_tune(model_name = "bert-tiny", dataset_path = "../fitness.json", lora_mode = "lora4", use_cached = True):
    """
    Fine-tune local Q&A dataset with LoRA. 

    Save fine-tuned model in 'ft/' for use when prompting, 
    including in-context learning.
    """

    if use_cached:
        model = utils.load_cached_model(model_name)
        tokenizer = utils.load_cached_tokenizer(model_name)
        dataset = utils.load_cached_dataset(dataset_path)
    else:
        model = utils.load_model(model_name)
        tokenizer = utils.load_tokenizer(model_name)
        dataset = utils.load_local_dataset(dataset_path)

    # FIXME: move to utils fn?
    inputs = tokenizer([qa['question'] for qa in dataset], 
                       max_length = 512,
                       return_tensors = 'pt', 
                       padding = True,
                       truncation = True)
    targets = tokenizer([qa['answer'] for qa in dataset],
                        max_length = 512,
                        return_tensors = 'pt',
                        padding = True,
                        truncation = True)

    # List of examples 
    x = inputs.input_ids
    y = targets.input_ids
    raw_x = [qa['question'] for qa in dataset]
    raw_y = [qa['answer'] for qa in dataset]

    # FIXME: remove debugger
    #pdb.set_trace()

    ######################################################################
    # FIXME: figure out which method to use...
    # Fine-tune with LoRA
    #lora_model = loraft.ft_bert(model, tokenizer, x, y, mode = lora_mode, debug = False)
    #lora_model = loraft.ft_bert(model, tokenizer, inputs, targets, mode = "all", debug = False)
    #lora_model = loraft.ft_bert(model, tokenizer, raw_x, raw_y, mode = "all", debug = False)
    #lora_model = loraft.ft_gen_bert(model, tokenizer, raw_x, raw_y, mode = "all", debug = False)
    
    #lora_model = loraft.hyper_lora(model, tokenizer, x, y, mode = lora_mode, debug = False)
    #lora_model = loraft.hyper_lora(model, tokenizer, x, y, mode = "all", debug = False)
    ######################################################################
    # FIXME: LangChain workflow
    data_chunks = langchainer()

    # FIXME: move to fns / file 
    template = """
        You are a personal trainer bot here to help answer clients' questions about health, fitness, exercise, and nutrition.

        Context: {context}
        --------------------
        History: {chat_history}
        --------------------
        Client: {question}
        Coach Copilot:
    """

    # FIXME: use QA chain 
    prompt = PromptTemplate(
        input_variables = ["chat_history", "context", "question"],
        template = template
    )

    # Conversation memory 
    memory = ConversationBufferMemory(memory_key = "chat_history",
                                      return_messages = True,
                                      input_key = "question")
    
    # QA retrival



    #print("\n ", data_chunks)

    # Save the fine-tuned model 
    """if not os.path.exists('ft'):
        os.makedirs('ft')
    lora_model.save_pretrained("ft/lora_tuned_model")

    print("Fine-tuning with LoRA complete. Model saved to 'ft/lora_tuned_model'")"""

def run_in_context_learning(model_name, prompt):
    # TODO: Load the fine-tuned model
    model = utils.load_model(model_name)
    tokenizer = utils.load_tokenizer(model_name)

    # TODO: Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

def chat():
    """
    Interact with chatbot!

    Type "exit" to exit
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = utils.generate_response(user_input)
        print("Coach Copilot (type 'exit' to stop): ", response)"""
    
    # FIXME: this is not where this would be...move!
    model_name = "bert-tiny"
    model = utils.load_cached_model(model_name)
    tokenizer = utils.load_cached_tokenizer(model_name)

    # FIXME: LangChain workflow
    #vectorstore = langchainer()
    vindex = langchainer()

    #pdb.set_trace()

    # FIXME: move to fns / file 
    template_ = """
        You are a personal trainer bot here to help answer clients' questions about health, fitness, exercise, and nutrition.

        Context: {context}
        --------------------
        History: {chat_history}
        --------------------
        Client: {question}
        Coach Copilot:
    """

    template = """
        You are a personal trainer bot named Coach Copilot. Help clients answer questions about health, fitness, exercise, and nutrition.

        Client: {question}
        -------------------
        Coach Copilot: 
    """

    # FIXME: use QA chain 
    prompt = PromptTemplate(
        #input_variables = ["chat_history", "context", "question"],
        input_variables = ["question"],
        template = template
    )

    # FIXME: make model 
    model_id = 'tiiuae/falcon-7b-instruct'

    falcon_llm = HuggingFaceHub(huggingfacehub_api_token = os.environ['HUGFACE_KEY'],
                            repo_id = model_id,
                            model_kwargs = {"temperature" : 0.8, 
                                            "max_new_tokens" : 2000})
    coach_copilot = LLMChain(
        #llm = model,
        llm = falcon_llm,
        prompt = prompt,
        verbose = True
    )

    # Conversation memory 
    """memory = ConversationBufferMemory(memory_key = "chat_history",
                                      return_messages = True,
                                      input_key = "question")"""
    
    # QA retrival
    """qa_chain = RetrievalQA.from_chain_type(
        llm = model,
        retriever = vindex.vectorstore.as_retriever(),
        chain_type = 'stuff',
        verbose = True,
        chain_type_kwargs = {
            "prompt": prompt,
            "memory": memory,
        }
    )"""

    # FIXME: remove debugger 
    #pdb.set_trace()

    while True:
        user_input = input("\n-------------------------------\nYou: ")
        if user_input.lower() == "exit":
            break
        #response = utils.generate_response(user_input)
        response = coach_copilot.run(user_input)
        print("Coach Copilot (type 'exit' to stop): ", response)

        print("\n+_+_+_+_+_+_+_+\n")
        print("Coach Copilot v2: ", vindex.query(user_input))
        print("\n")


def main():
    if args.task == 'run_ft':
        fine_tune(args.model, args.dataset, args.loramode)
    elif args.task == 'run_icl':
        run_in_context_learning(args.model, args.prompt)
    elif args.task == 'chat':
        chat()
    elif args.task == 'plot':
        utils.plot_results(args.output)  # Assuming you have a function for plotting
    elif args.task == 'cache':
        utils.cache_models_and_datasets()  # Assuming you have a function for caching

if __name__ == '__main__':
    main()
