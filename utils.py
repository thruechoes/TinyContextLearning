#from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import transformers as tf
import torch
import os
import json
#from datasets import load_dataset
import datasets
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# FIXME: remove debugger 
import pdb

CACHE_DIR = "./cache"

def get_hf_model_name(model: str) -> str:
    """
    Matches model name to the Hugging Face model name.
    """
    """return {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-med' : 'prajjwal1/bert-medium'
    }[model]"""
    # Returns input model name if not found 
    return {
        'bert-tiny' : 'prajjwal1/bert-tiny',
        #'bert-med'  : 'prajjwal1/bert-medium'
    }.get(model, model)

def load_model(model_name: str):
    """
    Load a pre-trained transformer model.

    Args:
    model_name (str): name or path of the model

    Returns:
    model: loaded transformer model
    """
    return tf.AutoModel.from_pretrained(model_name)

def load_model_for_classification(model_name: str):
    """
    Load a fine-tuned transformer model.
    
    Args:
    model_name (str): name or path of the model
    
    Returns:
    model: loaded transformer model
    """
    return tf.AutoModelForSequenceClassification.from_pretrained(model_name)

def load_tokenizer(model_name):
    """
    Load a tokenizer for a transformer model.
    
    Args:
    model_name (str): The name or path of the model.
    
    Returns:
    tokenizer: The tokenizer for the transformer model.
    """
    return tf.AutoTokenizer.from_pretrained(model_name)

def load_qa_dataset(file_path: str, n_train: int, n_val: int) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    questions = [pair['question'] for pair in data]
    answers = [pair['answer'] for pair in data]

    train_data = {'questions': questions[:n_train], 'answers': answers[:n_train]}
    val_data = {'questions': questions[n_train:n_train+n_val], 'answers': answers[n_train:n_train+n_val]}

    return train_data, val_data

def load_local_dataset(file_path: str):
    """
    Loads local JSON and imports as Huggingface Dataset object.
    """
    with open(file_path) as f:
        data = json.load(f)
    return datasets.Dataset.from_dict({"question": [d[0] for d in data],
                                       "answer"  : [d[1] for d in data]})

def preprocess_QA(examples):
    """
    Preprocess data before LoRA fine-tuning
    """
    """questions = [q.strip() for q in examples['question']]
    answers = [a.strip() for a in examples['answer']]
    inputs = tokenizer(questions, max_length=512, truncation=True, padding="max_length", return_tensors='pt')
    targets = tokenizer(answers, max_length=512, truncation=True, padding="max_length", return_tensors='pt').input_ids
    return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': targets}"""
    # FIXME: either remove or fix this function 
    return True
    

def process_data(file_path):
    """
    Process the Q&A data from a JSON file.
    
    Args:
    file_path (str): Path to the JSON file containing the Q&A pairs.
    
    Returns:
    processed_data: The processed data in a format suitable for training.
    """
    with open(file_path, 'r') as f:
        qa_pairs = json.load(f)
    
    # Assuming the format is a list of dictionaries with 'question' and 'answer' keys
    questions = [pair['question'] for pair in qa_pairs]
    answers = [pair['answer'] for pair in qa_pairs]
    
    # TODO: Add processing steps here (e.g., tokenization, converting labels to integers)
    
    return processed_data  # This should be your processed data

def cache_models_and_datasets():
    """
    Download and cache models and datasets.
    """
    # FIXME: remove debugger 
    #pdb.set_trace()

    # List of models to cache
    models = [
        'bert-tiny',
        #'bert-medium',
    ]
    
    # Download and cache models
    for model_name in models:
        hf_model_name = get_hf_model_name(model_name)
        print(f"Caching model: { hf_model_name }")
        model = load_model(hf_model_name)
        tokenizer = load_tokenizer(hf_model_name)
        
        # Cache 
        model.save_pretrained(os.path.join(CACHE_DIR, model_name))
        tokenizer.save_pretrained(os.path.join(CACHE_DIR, model_name))
    
    # List of datasets to cache (if any)
    # FIXME: arbitrarily using 35 / 16 split
    datasets = [
        {"filepath"    : "../fitness.json",
         "n_train"     : 35,
         "n_val"       : 16
        }
    ]
    
    # Download and cache datasets
    # FIXME: Remove caching of local dataset?
    for dataset in datasets:
        print(f"Caching dataset: {dataset}")
        with open(dataset['filepath'], 'r') as f:
            data = json.load(f)
        torch.save(data, os.path.join(CACHE_DIR, os.path.basename(dataset['filepath'])))
        #_ = load_qa_dataset(file_path = dataset['filepath'], n_train = dataset['n_train'], n_val = dataset['n_val'])
    
    print("Caching complete.")

########################################################################
# Caching functions 

def load_cached_model(model_name):
    return tf.AutoModel.from_pretrained(os.path.join(CACHE_DIR, model_name))

def load_cached_tokenizer(model_name):
    return tf.AutoTokenizer.from_pretrained(os.path.join(CACHE_DIR, model_name))

def load_cached_dataset(file_path):
    return torch.load(os.path.join(CACHE_DIR, os.path.basename(file_path)))

########################################################################

def plot_results(output_path):
    """
    Plot results (placeholder function).
    
    Args:
    output_path (str): Path to save the plot.
    """
    # TODO: Implement plotting logic here
    plt.plot([0, 1, 2], [0, 1, 4])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Results')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def save_model(model, output_dir):
    """
    Save a transformer model to a directory.
    
    Args:
    model: The transformer model to save.
    output_dir (str): Directory to save the model in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def train_model(model, tokenizer, data, output_dir):
    """
    Fine-tune a transformer model.
    
    Args:
    model: The transformer model to train.
    tokenizer: The tokenizer for the model.
    data: The training data.
    output_dir (str): Directory to save the fine-tuned model.
    
    Returns:
    model: The fine-tuned model.
    """
    # TODO: Preprocess the data if necessary
    
    # Define training arguments
    training_args = tf.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Define Trainer
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()

    # Save the fine-tuned model
    save_model(model, output_dir)

    return model

def generate_response(prompt):
    """
    Used to chat with Coach Copilot
    """
    model_name = "fine_tuned_model"
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    model = tf.AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors = "pt", max_length = 512, truncation = True)
    outputs = model.generate(**inputs, max_length = 512)
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return response

