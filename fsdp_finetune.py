# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import  DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from utils.save_utils import load_masked_model, load_masked_model_single
from utils.prehook_utils import put_backward_hooks, remove_all_hooks, check_whitelist
from utils.eval_utils import evaluate_and_log

from accelerate import Accelerator, DistributedType
from tqdm import tqdm

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#   - FSDP
#
# This example also demonstrates the checkpointing and sharding capabilities
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 32


# New Code #
# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# New Code #
# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *exc):
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


# For testing only
if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
    from accelerate.test_utils.training import mocked_dataloaders

    get_dataloaders = mocked_dataloaders  # noqa: F811


def training_function(config, args):
    # For testing only
    if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
        config["num_epochs"] = 2
    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="wandb", logging_dir=args.logging_dir
        )
    else:
        accelerator = Accelerator()
    accelerator.print(accelerator.distributed_type)

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("fsdp_glue_no_trainer", experiment_config)

    tokenizer = AutoTokenizer.from_pretrained(f'facebook/{config["model_name"]}', padding_side='left', model_max_length=512)
    #datasets = load_dataset("glue", "mrpc")
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    #tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        # Note: TPU support may vary by accelerate version; defaulting to 'longest' padding
        try:
            if accelerator.distributed_type == DistributedType.TPU:
                return tokenizer.pad(examples, padding="max_length", max_length=512, return_tensors="pt")
        except (AttributeError, Exception):
            pass
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"].with_format("torch"), collate_fn=collate_fn, batch_size=batch_size
    )

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    #model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, return_dict=True)
    if not config.get('model'):
        model = OPTForCausalLM.from_pretrained(f'facebook/{config["model_name"]}',
                                                          output_attentions=True,
                                                          output_hidden_states=True)

        # Only load pruned checkpoint if sparsity < 1.0
        if config["sparsity"] < 1.0:
            model = load_masked_model_single(model, f'pruned_models/{config["model_name"]}-{config["sparsity"]}.pt')
    else:
        model = config.get('model')
    

    # back_hooks = put_backward_hooks(model=model)

    # New Code #
    # For FSDP feature, it is highly recommended and efficient to prepare the model before creating optimizer
    model = accelerator.prepare(model)
    
    # Only register gradient masking hooks if sparsity < 1.0 (i.e., model is pruned)
    # IMPORTANT: Register hooks AFTER accelerator.prepare() to ensure mask is on the same device as gradients
    if config["sparsity"] < 1.0:
        for name, param in model.named_parameters():
            if 'weight' in name and check_whitelist(name):
                mask = (param.data != 0).to(param.device)  # Ensure mask is on same device as param
                print(f"prop nonzeros: {torch.sum(mask) / torch.numel(param)}")
                def hook(grad, mask=mask):
                    return grad * mask.float()
                param.register_hook(hook)
    #accelerator.print(model)

    # Instantiate optimizer
    # New Code #
    # For FSDP feature, at present it doesn't support multiple parameter groups,
    # so we need to create a single parameter group for the whole model
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=2e-4)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=2,
        num_training_steps=(config["train_steps"] * num_epochs) // gradient_accumulation_steps,
    )

    # New Code #
    # For FSDP feature, prepare everything except the model as we have already prepared the model
    # before creating the optimizer
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    overall_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            num_epochs -= int(training_difference.replace("epoch_", ""))
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            num_epochs -= resume_step // len(train_dataloader)
            # If resuming by step, we also need to know exactly how far into the DataLoader we went
            resume_step = (num_epochs * len(train_dataloader)) - resume_step

    # Evaluate TWO baselines: unpruned model and pruned model (before fine-tuning)
    perplexity_history = {
        'unpruned_baseline': None,
        'pruned_baseline': None,
        'epochs': []
    }
    
    # 1. Evaluate unpruned model baseline (if we have a pruned model)
    if config["sparsity"] < 1.0:
        accelerator.print("\n🔍 Evaluating UNPRUNED baseline model...")
        accelerator.print("  (Loading original model without pruning)")
        
        # Create a fresh unpruned model
        unpruned_model = OPTForCausalLM.from_pretrained(
            f'facebook/{config["model_name"]}',
            output_attentions=True,
            output_hidden_states=True
        )
        unpruned_model = accelerator.prepare(unpruned_model)
        
        unpruned_ppl = evaluate_and_log(
            model=unpruned_model,
            model_name=config["model_name"],
            epoch=None,
            config={**config, 'sparsity': 1.0},  # Mark as unpruned
            device=accelerator.device,
            accelerator=accelerator
        )
        perplexity_history['unpruned_baseline'] = unpruned_ppl
        
        # Clean up unpruned model
        del unpruned_model
        torch.cuda.empty_cache()
        
        accelerator.print("\n🔍 Evaluating PRUNED baseline model (before fine-tuning)...")
    else:
        accelerator.print("\n🔍 Evaluating baseline model...")
    
    # 2. Evaluate pruned model baseline (or unpruned if sparsity=1.0)
    pruned_ppl = evaluate_and_log(
        model=model,
        model_name=config["model_name"],
        epoch=None,
        config=config,
        device=accelerator.device,
        accelerator=accelerator
    )
    perplexity_history['pruned_baseline'] = pruned_ppl
    
    # Print comparison if we have both baselines
    if perplexity_history['unpruned_baseline'] is not None:
        degradation = ((pruned_ppl - unpruned_ppl) / unpruned_ppl) * 100
        accelerator.print("\n" + "="*70)
        accelerator.print("📊 Baseline Comparison")
        accelerator.print("="*70)
        accelerator.print(f"  🟢 Unpruned Model: {unpruned_ppl:.2f}")
        accelerator.print(f"  🔴 Pruned Model (before training): {pruned_ppl:.2f}")
        accelerator.print(f"  📉 Degradation: +{degradation:.1f}%")
        accelerator.print(f"  🎯 Goal: Recover performance through fine-tuning")
        accelerator.print("="*70)

    # Now we train the model
    if not config['resume']:
        for epoch in range(num_epochs):
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"📊 Epoch {epoch + 1}/{num_epochs}")
            accelerator.print(f"{'='*70}")

            # New Code #
            # context manager to track the peak memory usage during the training epoch
            with TorchTracemalloc() as tracemalloc:
                model.train()
                if args.with_tracking:
                    total_loss = 0

                epoch_loss = 0
                num_batches = 0

                # Create progress bar for steps within epoch
                progress_bar = tqdm(
                    enumerate(train_dataloader),
                    total=config['max_step'],
                    desc=f"Epoch {epoch + 1}",
                    disable=not accelerator.is_local_main_process
                )

                for step, batch in progress_bar:
                    if step == config['max_step']:
                        break
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == 0:
                        if resume_step is not None and step < resume_step:
                            pass
                    # We could avoid this line since we set the accelerator with    `device_placement=True`.
                    batch.to(accelerator.device)
                    outputs = model(**batch, labels=batch['input_ids'])
                    # print(f"max memory: {torch.cuda.memory_allocated()}")
                    loss = outputs.loss
                    #print(f'Loss: {loss}')
                    loss = loss / gradient_accumulation_steps
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()

                    epoch_loss += loss.detach().float()
                    num_batches += 1

                    # Update progress bar with current loss
                    current_avg_loss = (epoch_loss / num_batches).item()
                    progress_bar.set_postfix({
                        'loss': f'{current_avg_loss:.4f}',
                        'step': f'{step}/{config["max_step"]}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })

                    accelerator.backward(loss)
                    if step % gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        # accelerator.print(lr_scheduler.get_lr())

                    overall_step += 1

                    # # Log every 100 steps
                    # if step % 100 == 0 and step > 0:
                    #     accelerator.print(
                    #         f"  📈 Step {step}/{config['max_step']} | "
                    #         f"Loss: {current_avg_loss:.4f} | "
                    #         f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    #     )

                    if isinstance(checkpointing_steps, int):
                        output_dir = f"step_{overall_step}"
                        if overall_step % checkpointing_steps == 0:
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)

                # Close progress bar
                progress_bar.close()

                # Epoch summary (Note: memory stats will be printed after context manager   exits)
                final_avg_loss = (epoch_loss / num_batches)
                accelerator.print(f"\n{'='*70}")
                accelerator.print(f"✅ Epoch {epoch + 1} Complete")
                accelerator.print(f"{'='*70}")
                accelerator.print(f"  📊 Average Loss: {final_avg_loss:.4f}")
                accelerator.print(f"  🔢 Total Steps: {num_batches}")

            # New Code #
            # Printing the GPU memory usage details such as allocated memory, peak memory,  and total memory usage
            # (This prints AFTER the context manager exits, so tracemalloc.peaked is now    available)
            accelerator.print(f"\n💾 Memory Stats:")
            accelerator.print(f"  Memory before entering the train: {b2mb(tracemalloc.begin)}   MB")
            accelerator.print(f"  Memory consumed (end-begin): {tracemalloc.used} MB")
            accelerator.print(f"  Peak Memory consumed (max-begin): {tracemalloc.peaked} MB")
            accelerator.print(
                f"  Total Peak Memory (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}     MB"
            )
            # Logging the peak memory usage of the GPU to the tracker
            if args.with_tracking:
                accelerator.log(
                    {
                        "train_total_peak_memory": tracemalloc.peaked + b2mb(tracemalloc.   begin),
                    },
                    step=epoch,
            )

            # Evaluate perplexity after this epoch
            accelerator.print(f"\n🔍 Evaluating model after Epoch {epoch + 1}...")
            epoch_ppl = evaluate_and_log(
                model=model,
                model_name=config["model_name"],
                epoch=epoch,
                config=config,
                device=accelerator.device,
                accelerator=accelerator
            )

            # Store perplexity for this epoch
            perplexity_history['epochs'].append({
                'epoch': epoch + 1,
                'perplexity': epoch_ppl,
                'avg_loss': final_avg_loss.item() if hasattr(final_avg_loss, 'item') else   final_avg_loss
            })

            # Print perplexity trend
            accelerator.print(f"\n📈 Perplexity Trend:")
            if perplexity_history['unpruned_baseline'] is not None:
                accelerator.print(f"  🟢 Unpruned Baseline: {perplexity_history ['unpruned_baseline']:.2f}")
            accelerator.print(f"  🔴 Pruned Baseline: {perplexity_history['pruned_baseline']:.  2f}")
            for ep_data in perplexity_history['epochs']:
                improvement = ((perplexity_history['pruned_baseline'] - ep_data ['perplexity']) / perplexity_history['pruned_baseline']) * 100
                accelerator.print(
                    f"  🔵 Epoch {ep_data['epoch']}: {ep_data['perplexity']:.2f} "
                    f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}% vs pruned)"
                )
        if not config.get('save_model') or config['save_model']:
            # Create pruned_models directory if it doesn't exist
            os.makedirs('pruned_models', exist_ok=True)
            # Save state_dict instead of full model for better compatibility
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f'pruned_models/{config["model_name"]}-{config["sparsity"]}-stage1.pt')
            accelerator.print(f"✅ Model state_dict saved to pruned_models/{config['model_name']}-{config['sparsity']}-stage1.pt")
    else:
        ## load model from stage 1,f'pruned_models/{config["model_name"]}-{config["sparsity"]}-stage1.pt'
        accelerator.print(f"\n===== 🔄 Resuming from checkpoint for Stage 2 LoRA Fine-tuning =====")
        model = load_masked_model_single(model, f'pruned_models/{config["model_name"]}-{config["sparsity"]}-stage1.pt')
        model = accelerator.prepare(model)
    if config.get("use_lora_per_layer", False):
        accelerator.print(f"\n===== 🎯 Stage 2: Injecting LoRA + Fine-tuning for {config['lora_epochs']} epochs =====")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        accelerator.print("✅ Injected LoRA adapters in all layers.")
    
        # 可选：解冻主干
        if config.get("unfreeze_base", False):
            for p in model.parameters():
                p.requires_grad = True
            accelerator.print("🔓 Unfroze base parameters (train LoRA + backbone).")
        else:
            accelerator.print("🧊 Frozen base model (train LoRA only).")
    
        # prepare LoRA model
        model = accelerator.prepare(model)
    
        # 重新定义优化器和 scheduler (dataloader不需要重新prepare)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=2e-4)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=2,
            num_training_steps=(config["train_steps"] * config["lora_epochs"])
        )
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    
        # 训练阶段 2
        for epoch in range(config.get("lora_epochs", 5)):
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"🎯 LoRA Fine-tune Epoch {epoch + 1}/{config.get('lora_epochs', 5)}")
            accelerator.print(f"{'='*70}")

            model.train()
            epoch_loss = 0
            num_batches = 0

            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=config['max_step'],
                desc=f"LoRA Epoch {epoch + 1}",
                disable=not accelerator.is_local_main_process
            )

            for step, batch in progress_bar:
                if step == config['max_step']:
                    break
                    
                # Batch is already prepared by accelerator, just move to device
                batch.to(accelerator.device)
                
                # Forward pass with labels (same as Stage 1)
                outputs = model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                
                # Track loss before backward
                epoch_loss += loss.detach().float()
                num_batches += 1
                
                # Backward and optimize
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update progress bar
                current_avg_loss = (epoch_loss / num_batches).item()
                progress_bar.set_postfix({
                    'loss': f'{current_avg_loss:.4f}',
                    'step': f'{step+1}/{config["max_step"]}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            progress_bar.close()
            final_avg_loss = (epoch_loss / num_batches).item()
            accelerator.print(f"\n✅ LoRA Epoch {epoch + 1} Complete | Avg Loss: {final_avg_loss:.4f}")

            # Evaluate perplexity after each LoRA epoch
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"📊 Evaluating Perplexity after LoRA Epoch {epoch + 1}")
            accelerator.print(f"{'='*70}")
            
            # For LoRA stage, we offset the epoch number to continue from Stage 1
            stage2_epoch = num_epochs + epoch  # Continue epoch numbering from Stage 1
            evaluate_and_log(model, config["model_name"], stage2_epoch, config, accelerator.device, accelerator)
            
        accelerator.print("\n🎯 Two-stage fine-tuning completed successfully.")
    remove_all_hooks(model)
    torch.cuda.empty_cache()
    
    # Print final summary
    accelerator.print("\n" + "="*70)
    accelerator.print("📊 Final Perplexity Summary")
    accelerator.print("="*70)
    
    if perplexity_history['unpruned_baseline'] is not None:
        accelerator.print(f"  🟢 Unpruned Baseline: {perplexity_history['unpruned_baseline']:.2f}")
    accelerator.print(f"  🔴 Pruned Baseline (before training): {perplexity_history['pruned_baseline']:.2f}")
    
    if perplexity_history['epochs']:
        best_epoch = min(perplexity_history['epochs'], key=lambda x: x['perplexity'])
        final_epoch = perplexity_history['epochs'][-1]
        accelerator.print(f"  🔵 Final (Epoch {final_epoch['epoch']}): {final_epoch['perplexity']:.2f}")
        accelerator.print(f"  ⭐ Best (Epoch {best_epoch['epoch']}): {best_epoch['perplexity']:.2f}")
        
        # Calculate improvements
        improvement_vs_pruned = ((perplexity_history['pruned_baseline'] - final_epoch['perplexity']) / perplexity_history['pruned_baseline']) * 100
        accelerator.print(f"  📈 Improvement vs Pruned: {improvement_vs_pruned:+.1f}%")
        
        if perplexity_history['unpruned_baseline'] is not None:
            recovery_rate = ((perplexity_history['unpruned_baseline'] - final_epoch['perplexity']) / 
                           (perplexity_history['unpruned_baseline'] - perplexity_history['pruned_baseline'])) * 100
            gap_remaining = final_epoch['perplexity'] - perplexity_history['unpruned_baseline']
            accelerator.print(f"  🎯 Recovery Rate: {recovery_rate:.1f}%")
            accelerator.print(f"  📊 Gap to Unpruned: {gap_remaining:+.2f} perplexity points")
    
    accelerator.print("="*70)

    if not config.get('save_model') or config['save_model']:
        # Create pruned_models directory if it doesn't exist
        os.makedirs('pruned_models', exist_ok=True)
        torch.save(model, f'pruned_models/{config["model_name"]}-{config["sparsity"]}-finetuned.pt')
        accelerator.print(f"Model saved to pruned_models/{config['model_name']}-{config['sparsity']}-finetuned.pt")


def fsdp_finetune(config):
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    #args = parser.parse_args()
    args = parser.parse_args(['--mixed_precision', 'fp16'])
    training_function(config, args)