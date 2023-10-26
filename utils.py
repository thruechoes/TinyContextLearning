#from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import transformers as tf
import torch
import os
import json
#from datasets import load_dataset
import datasets
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

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

def load_model(model_name):
    """
    Load a pre-trained or fine-tuned transformer model.
    
    Args:
    model_name (str): The name or path of the model.
    
    Returns:
    model: The loaded transformer model.
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
    # List of models to cache
    models = [
        'bert-tiny',
        #'bert-medium',
    ]
    
    # Download and cache models
    for model in models:
        hf_model_name = get_hf_model_name(model)
        print(f"Caching model: { hf_model_name }")
        _ = load_model(hf_model_name)
        _ = load_tokenizer(hf_model_name)
    
    # List of datasets to cache (if any)
    datasets = [
        {"filepath"    : "../fitness.json",
         "n_train"     : 20,
         "n_val"       : 10
        }
    ]
    
    # Download and cache datasets
    for dataset in datasets:
        print(f"Caching dataset: {dataset}")
        _ = load_qa_dataset(file_path = dataset['filepath'], n_train = dataset['n_train'], n_val = dataset['n_val'])
    
    print("Caching complete.")

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
