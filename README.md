# Tiny Context Learning

Using Local and Global Entropy to order few-shot prompts. 

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
