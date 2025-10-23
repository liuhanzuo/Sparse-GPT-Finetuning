"""
Evaluation utilities for measuring model perplexity on Wikitext dataset.
Based on Testing.ipynb implementation.
"""

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def evaluate_perplexity(model, model_name, device='cuda', token_length=2048, stride=2048, accelerator=None):
    """
    Evaluate model perplexity on Wikitext-2 test set.
    
    Args:
        model: The model to evaluate (should be on correct device)
        model_name: Name of the model (for tokenizer)
        device: Device to run evaluation on
        token_length: Maximum sequence length
        stride: Stride for sliding window
        accelerator: Optional Accelerate accelerator for distributed evaluation
        
    Returns:
        float: Perplexity score
    """
    
    def print_fn(msg):
        """Helper to print via accelerator if available"""
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        f'facebook/{model_name}', 
        padding_side='left',
        use_fast=False
    )
    
    # Load Wikitext-2 test set
    print_fn("  📖 Loading Wikitext-2 test set...")
    test_set = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer("\n\n".join(test_set['text']), return_tensors='pt')
    
    # Set model to eval mode
    model.eval()
    
    seq_len = encodings.input_ids.size(1)
    print_fn(f"  📏 Sequence length: {seq_len}")
    
    nlls = []
    prev_end_loc = 0
    
    # Create progress bar
    progress_bar = tqdm(
        range(0, seq_len, stride),
        desc="  🧮 Computing perplexity",
        disable=not (accelerator is None or accelerator.is_local_main_process)
    )
    
    for begin_loc in progress_bar:
        end_loc = min(begin_loc + token_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device=device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    progress_bar.close()
    
    # Set model back to train mode
    model.train()
    
    return ppl.item()


def evaluate_and_log(model, model_name, epoch, config, device='cuda', accelerator=None):
    """
    Evaluate model and log results.
    
    Args:
        model: Model to evaluate
        model_name: Name of the model
        epoch: Current epoch number (None for baseline)
        config: Training configuration dict
        device: Device to run on
        accelerator: Optional Accelerate accelerator
        
    Returns:
        float: Perplexity score
    """
    
    def print_fn(msg):
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)
    
    if epoch is None:
        print_fn("\n" + "="*70)
        print_fn("📊 Baseline Evaluation (Before Training)")
        print_fn("="*70)
    else:
        print_fn("\n" + "="*70)
        print_fn(f"📊 Evaluation After Epoch {epoch + 1}")
        print_fn("="*70)
    
    try:
        ppl = evaluate_perplexity(
            model=model,
            model_name=model_name,
            device=device,
            token_length=2048,
            stride=2048,
            accelerator=accelerator
        )
        
        if epoch is None:
            print_fn(f"  ✅ Baseline Perplexity: {ppl:.2f}")
        else:
            print_fn(f"  ✅ Perplexity: {ppl:.2f}")
            print_fn(f"  📈 Epoch: {epoch + 1}/{config['num_epochs']}")
            print_fn(f"  🎯 Sparsity: {config['sparsity']}")
        
        print_fn("="*70)
        
        return ppl
        
    except Exception as e:
        print_fn(f"  ❌ Evaluation failed: {str(e)}")
        print_fn("="*70)
        return None
