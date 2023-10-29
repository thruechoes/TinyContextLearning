import torch
import transformers
import torch.nn as nn
import numpy as np
import os
from typing import List
import tqdm
import copy

# FIXME: remove debugger 
import pdb

# LoRA Config: Param Efficient Fine Tuning
import peft

######################################################################
# DETERMINE TYPE OF DEVICE AUTOMATICALLY
DEVICE = os.environ.get("DEVICE", "cpu")

if DEVICE == "gpu":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

print("Low-Rank Approximation using: ", DEVICE)
######################################################################

class LoRA(nn.Module):
    """
    LoRA - Low-Rank Approximation

    FIXME: add args and return 
    """
    def __init__(self, linear_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = linear_module

        # Get original weight matrix W 
        d1, d2 = self.base_module.weight.shape

        # Initialize LoRA matrices with a small scale for B
        scale = 0.02
        self.lora_A = nn.Parameter(torch.randn(d1, lora_rank) * scale)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, d2) * scale)

        # Freeze pre-trained model weights
        for param in self.base_module.parameters():
            param.requires_grad = False

    def forward(self, x):
        # FIXME: v1 
        # Compute original linear transformation
        original_conv = self.base_module(x)
        """
        # Compute low-rank updatew
        lora_out = torch.mm(torch.mm(x, self.lora_A), self.lora_B)

        return original_conv + lora_out"""
        # FIXME: v2 
        """A_BT = torch.matmul(self.lora_A, self.lora_B.t())
        x_A_BT = torch.matmul(x.transpose(-1, -1), A_BT)
        return original_conv + x_A_BT"""

        # FIXME: remove debugger 
        #pdb.set_trace()
    
        # FIXME: v3
        A_x = torch.matmul(x.transpose(-1, -1), self.lora_A)
        lora_out = torch.matmul(A_x, self.lora_B)  # or self.lora_B.t()?
        return original_conv + lora_out

    



def parameters_to_fine_tune(model: nn.Module, mode: str) -> List[nn.Parameter]:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    FIXME: add args and return 
    """
    # Get all LoRA parameters
    lora_params = [p for m in model.modules() if isinstance(m, LoRA) for p in m.parameters()]

    if mode == 'all':
        return list(model.parameters())
    elif mode == 'last':
        return list(model.transformer.h[-1].parameters()) + lora_params
    elif mode == 'first':
        return list(model.transformer.h[0].parameters()) + lora_params
    elif mode == 'middle':
        nblocks = len(model.transformer.h)
        mid_idx = nblocks // 2
        return list(model.transformer.h[mid_idx].parameters()) + lora_params
    elif mode.startswith('lora'):
        return lora_params
    else:
        raise NotImplementedError()

def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss.

    FIXME: add args and return 
    """
    if logits.dim() == 2:
        return nn.CrossEntropyLoss()(logits, targets)
    elif logits.dim() == 3:
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        return nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.transpose(1, 2), shift_targets)
    else:
        raise ValueError('Invalid logits dimensions')

def get_acc(logits, targets):
    """
    Computes the exact match accuracy.

    FIXME: add args and return
    """
    if logits.dim() == 2:
        predicts = logits.argmax(dim=1)
        return (predicts == targets).float().mean().item()
    elif logits.dim() == 3:
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        predicts = shift_logits.argmax(dim=2)
        mask = shift_targets != -100
        correct = (predicts == shift_targets) & mask
        return correct.float().mean().item()
    else:
        raise ValueError('Invalid logits dimensions')

def ft_bert(model, tok, x, y, mode, debug, batch_size=8):
    """
    FIXME: Add docstring 
    """
    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for layer in model.transformer.h:
            layer.mlp.c_fc = LoRA(layer.mlp.c_fc, int(mode[4:]))
            layer.mlp.c_proj = LoRA(layer.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(x, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok([x[i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if debug:
            break

        if step % 10 == 0:
            with torch.no_grad():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    return model

def eval(model: nn.Module, tok: transformers.PreTrainedTokenizer, val_data: dict) -> float:
    """
    Evaluate the model on validation data.

    NOTE: logits are unnormalized predictions outputted by the model 

    Args:
    model (nn.Module): The pre-trained transformer model fine-tuned for a specific task.
    tok (transformers.PreTrainedTokenizer): The tokenizer that was used during model fine-tuning.
    val_data (dict): A dictionary containing the validation data. It has two keys:
                     'x' which maps to the text data, and 'y' which maps to the labels.

    Returns:
    float: The accuracy of the model on the validation data.
    """
    # Tokenize the validation text data and convert it to a format that can be processed by the model.
    # Ensure that the tokenized data is padded to the same length, truncated if necessary, and 
    # moved to the same device as the model.
    x = tok(val_data['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)

    # Convert the validation labels to a tensor and move them to the same device as the model.
    y = torch.tensor(val_data['y'], device=DEVICE)

    # Disable gradient calculation since we are in evaluation mode.
    # This reduces memory usage and speeds up computation.
    with torch.inference_mode():
        # Pass the tokenized text data through the model to get the logits.
        logits = model(**x).logits

        # Calculate and return the accuracy of the model on the validation data.
        return get_acc(logits, y)

def hyper_lora(model, tok, x, y, mode, debug, batch_size = 8, all_layers = False):
    """
    Hyperparameters of LoRA
    """
    if all_layers:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
    else:
        # Only target attention blocks of model 
        target_modules = ["q_proj", "v_proj"]
    
    """lora_config = peft.LoraConfig(
        r = 8,   # or r = 16
        target_modules = target_modules,
        lora_alpha = 8,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
    )"""

    model = copy.deepcopy(model).to(DEVICE)

    if mode.startswith('lora'):
        for layer in model.encoder.layer:
            layer.attention.self.query = LoRA(layer.attention.self.query, int(mode[4:]))
            layer.attention.self.key = LoRA(layer.attention.self.key, int(mode[4:]))
            layer.attention.self.value = LoRA(layer.attention.self.value, int(mode[4:]))
            layer.attention.output.dense = LoRA(layer.attention.output.dense, int(mode[4:]))
            layer.intermediate.dense = LoRA(layer.intermediate.dense, int(mode[4:]))
            layer.output.dense = LoRA(layer.output.dense, int(mode[4:]))
    else:
        print("\nNot using LoRA for Fine-Tuning\n")

    optimizer = torch.optim.AdamW(parameters_to_fine_tune(model, mode), lr=5e-5)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # FIXME: remove debugger 
    #pdb.set_trace()

    for epoch in range(3):  # Number of training epochs
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(loader):
            input_ids, labels = [b.to(DEVICE) for b in batch]

            outputs = model(input_ids)
            logits = outputs.logits
            loss = get_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

        if debug:
            break

    return model
    
def get_gen_loss(logits, labels):
    """
    Calculate the cross-entropy loss between the predicted logits and the ground truth labels.

    Args:
    logits (torch.Tensor): The predicted logits from the model of shape (batch_size, sequence_length, vocab_size).
    labels (torch.Tensor): The ground truth token IDs of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: The calculated loss.
    """
    # Flatten the tensors to make them compatible for cross entropy calculation
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    # Calculate the cross entropy loss
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

    return loss

def get_gen_acc(logits, labels):
    """
    Calculate the accuracy of the predicted logits compared to the ground truth labels.

    Args:
    logits (torch.Tensor): The predicted logits from the model of shape (batch_size, sequence_length, vocab_size).
    labels (torch.Tensor): The ground truth token IDs of shape (batch_size, sequence_length).

    Returns:
    float: The calculated accuracy.
    """
    # Get the predicted token IDs from logits
    preds = torch.argmax(logits, dim=-1)
    
    # Create a mask for non-pad tokens (labels not equal to -100)
    mask = (labels != -100)
    
    # Calculate the number of correct predictions and the total number of predictions
    num_correct = ((preds == labels) & mask).sum().item()
    num_total = mask.sum().item()
    
    # Calculate the accuracy
    accuracy = num_correct / num_total if num_total > 0 else 0

    return accuracy

def ft_gen_bert(model, tok, x, y, mode, debug, batch_size=8):
    """
    Fine-tunes a BERT model on a local Q&A dataset for a text generation task.

    Args:
    model (transformers.PreTrainedModel): The BERT model to be fine-tuned.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the BERT model.
    x (list): List of tokenized questions.
    y (list): List of tokenized answers.
    mode (str): Mode for fine-tuning. If it starts with 'lora', LoRA layers are applied.
    debug (bool): If True, run the function in debug mode (quick run for testing).
    batch_size (int, optional): Size of the batches for training. Defaults to 8.

    Returns:
    transformers.PreTrainedModel: The fine-tuned BERT model.
    """
    # Copy the model to avoid changes to the original model
    model = copy.deepcopy(model)
    
    # Check if LoRA should be applied
    if mode.startswith('lora'):
        for layer in model.transformer.h:
            layer.mlp.c_fc = LoRA(layer.mlp.c_fc, int(mode[4:]))
            layer.mlp.c_proj = LoRA(layer.mlp.c_proj, int(mode[4:]))
    
    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prepare the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # FIXME: remove debugger 
    #pdb.set_trace()
    
    # Prepare the data
    #all_x = tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    all_x = tok(x, return_tensors = 'pt', padding = True, truncation = True, max_length = 512).to(DEVICE)
    all_y = torch.tensor(y, device=device)
    #all_x = x
    #all_y = y
    
    # Set the model to training mode
    #model.train()
    
    # Initialize progress bar
    num_steps = 1000 if not debug else 10  # Reduce number of steps if in debug mode
    pbar = tqdm.tqdm(range(num_steps))
    
    for step in pbar:
        # FIXME: remove debugger 
        pdb.set_trace()

        # Randomly sample a batch of data
        batch_indices = np.random.randint(0, len(x), batch_size)
        batch_x = {k: v[batch_indices] for k, v in all_x.items()}
        batch_y = all_y[batch_indices]
        
        # Forward pass
        model.train()
        outputs = model(**batch_x, labels=batch_y)
        loss = outputs.loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Break the loop in debug mode after the first iteration
        if debug:
            break

        # Evaluate model every 10 steps
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(**all_x).logits
            acc = get_gen_acc(logits, all_y)
            
            # Update progress bar
            pbar.set_description(f'Loss: {loss.item():.04f}, Acc: {acc:.04f}')

            # Early cutoff if accuracy high enough
            if acc > 0.75:
                break

    # Set the model back to evaluation mode
    #model.eval()
    
    return model
