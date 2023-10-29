# Tiny Context Learning

Using Local and Global Entropy to order few-shot prompts. 

## QuickStart 

The following steps can be implemented with this codebase: 

### Cache Model and Local Dataset 

```
python main.py --task cache --model bert-tiny --dataset path/to/local_data.json
```

### Fine-tune model with LoRA 

The *--loramode* parameter takes the form **loraN** where N can take the values **[2, 4, 8, 16]**.

E.g. 

```
python main.py --task run_ft --model bert-tiny --dataset path/to/local_data.json --loramode lora4
```

## Query Splitting 

Split prompt into types of knowledge, run subtypes through LLM, and combine results into one. 

### 1. Define Knowledge Types 

### 2. Prompt Splitter 

Either classifier or heuristic to determine how to split a prompt. 

E.g. key-phrasing: "...get a six pack..." or "...get abs..." -> nutrition = 0.8

### 3. Type Queries 

Each split (type) goes through LLM. 

### 4. Aggregate Reponses 

Combine all of the responses into one. 
