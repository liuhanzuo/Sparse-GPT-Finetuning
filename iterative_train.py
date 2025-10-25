import os
import gc
import json
import argparse
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from torch.nn.utils import prune

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator, DistributedType

# =============================
# Constants / Utils
# =============================
MAX_GPU_BATCH_SIZE = 32
OPT_BLACKLIST = ['model.decoder.embed_tokens', 'model.decoder.embed_positions']


def b2mb(x):  # bytes -> MB
    return int(x / 2**20)


class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            self.begin = torch.cuda.memory_allocated()
        else:
            self.begin = 0
        return self

    def __exit__(self, *exc):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
            self.used = b2mb(self.end - self.begin)
            self.peaked = b2mb(self.peak - self.begin)
        else:
            self.end = self.peak = self.used = self.peaked = 0


def unwrap(accelerator, model):
    try:
        return accelerator.unwrap_model(model)
    except Exception:
        return model


def is_pruned_module(m):
    # 被 PyTorch prune 后的模块会有 weight_orig / weight_mask
    return hasattr(m, "weight_mask") and hasattr(m, "weight_orig")


def pruned_param_name_set(model):
    names = set()
    for mod_name, m in model.named_modules():
        if is_pruned_module(m):
            names.add(f"{mod_name}.weight_orig")
            if hasattr(m, "weight"):
                names.add(f"{mod_name}.weight")
    return names


def snapshot_params(model):
    snap = {}
    for n, p in model.named_parameters():
        snap[n] = p.detach().float().cpu().clone()
    return snap


def diff_report(before, after, pruned_names, eps=1e-7):
    changed_nonpruned, changed_pruned = [], []
    for n, t0 in before.items():
        t1 = after[n]
        if t0.shape != t1.shape:
            changed_nonpruned.append((n, "shape-changed"))
            continue
        delta = (t0 - t1).abs().max().item()
        if delta > eps:
            (changed_pruned if n in pruned_names else changed_nonpruned).append((n, delta))
    return changed_pruned, changed_nonpruned


def calculate_sparsity(model, module_blacklist=OPT_BLACKLIST):
    total_params, zero_params = 0, 0
    for name, param in model.named_parameters():
        if any(b in name for b in module_blacklist):
            continue
        if 'weight' not in name or len(param.shape) < 2:
            continue
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    return 0.0 if total_params == 0 else (zero_params / total_params) * 100.0


def freeze_non_pruned_params(model):
    """
    冻结所有非剪枝层（包括 bias / LN / embedding / 输出层）
    只保留被剪枝的主干线性层 weight 可训练
    """
    count_trainable = 0
    for name, p in model.named_parameters():
        if any(x in name for x in [
            "q_proj.weight", "k_proj.weight", "v_proj.weight",
            "out_proj.weight", "fc1.weight", "fc2.weight"
        ]):
            p.requires_grad = True
            count_trainable += 1
        else:
            p.requires_grad = False
    print(f"🧊 已冻结非剪枝参数，仅保留 {count_trainable} 个可训练层。")


def get_trainable_params(model):
    # 仅返回 requires_grad=True 的参数
    return [p for p in model.parameters() if p.requires_grad]


def prune_percentage_of_nonzeros(model, percentage=0.1, module_blacklist=OPT_BLACKLIST):
    """
    每次根据当前非零权重，按 L1 全局裁掉一定比例。
    """
    parameters_to_prune = []
    module_dict = {n: m for n, m in model.named_modules()}

    for name, param in model.named_parameters():
        if any(b in name for b in module_blacklist):
            continue
        if 'weight' not in name or len(param.shape) < 2:
            continue

        if name.endswith('.weight'):
            module_name = name[:-7]
        elif name.endswith('.weight_orig'):
            module_name = name[:-12]
        else:
            continue

        if module_name in module_dict:
            parameters_to_prune.append((module_dict[module_name], 'weight'))

    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=percentage,
        )

    return len(parameters_to_prune)


@torch.no_grad()
def apply_mask(model):
    """
    硬掩码：确保被剪掉的权重不复活。
    """
    for _, module in model.named_modules():
        if hasattr(module, "weight_mask"):
            module.weight_orig.data.mul_(module.weight_mask.data)


# =============================
# Training (iterative pruning)
# =============================
def iterative_training(config, args):
    """
    迭代式剪枝训练：
      - 先评估基线
      - 每次裁剪当前非零权重的 p%（全局 L1）
      - 只训练被剪枝的线性层，其他全部冻结
      - 每步更新后应用硬掩码，避免零权重复活
      - 使用 label masking、梯度裁剪，稳定 PPL
    """
    # accelerate
    if os.environ.get("TESTING_MOCKED_DATALOADERS", None) == "1":
        config["num_epochs"] = 2

    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision=args.mixed_precision
    ) if args.with_tracking else Accelerator()

    accelerator.print(accelerator.distributed_type)

    # 训练超参
    lr_cfg = config["lr"]
    # 稀疏阶段 LR 限制（关键！避免发散）
    lr_sparse = min(lr_cfg, 2e-6)

    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    # 数据
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        f'facebook/{config["model_name"]}',
        padding_side='left',
        model_max_length=512
    )
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

    def collate_fn(examples):
        try:
            if accelerator.distributed_type == DistributedType.TPU:
                return tokenizer.pad(examples, padding="max_length", max_length=512, return_tensors="pt")
        except Exception:
            pass
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        tokenized_datasets["train"].with_format("torch"),
        collate_fn=collate_fn,
        batch_size=batch_size
    )

    # 模型
    model = OPTForCausalLM.from_pretrained(
        f'facebook/{config["model_name"]}',
        output_attentions=True,
        output_hidden_states=True
    )
    model = accelerator.prepare(model)

    # 仅为保持接口一致，这里先创建一个“临时”的 opt/scheduler；真实训练时每个迭代会重建
    tmp_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_sparse, weight_decay=2e-4)
    tmp_scheduler = get_linear_schedule_with_warmup(
        optimizer=tmp_optimizer,
        num_warmup_steps=2,
        num_training_steps=(config["max_steps"] * num_epochs) // gradient_accumulation_steps,
    )
    tmp_optimizer, train_dataloader, tmp_scheduler = accelerator.prepare(
        tmp_optimizer, train_dataloader, tmp_scheduler
    )

    # 迭代式剪枝配置
    target_sparsity = config["target_sparsity"]      # e.g. 0.9
    prune_percentage = config.get("prune_percentage", 0.1)
    epochs_per_iteration = config.get("epochs_per_iteration", 3)
    max_iterations = config.get("max_iterations", 20)

    accelerator.print("\n" + "="*70)
    accelerator.print("🔄 ITERATIVE PRUNING TRAINING")
    accelerator.print("="*70)
    accelerator.print(f"  🎯 Target Sparsity: {target_sparsity*100:.1f}%")
    accelerator.print(f"  ✂️ Prune per iteration: {prune_percentage*100:.1f}% of non-zeros")
    accelerator.print(f"  📚 Epochs per iteration: {epochs_per_iteration}")
    accelerator.print("="*70)

    # 基线 PPL
    perplexity_history = {'unpruned_baseline': None, 'iterations': []}

    # —— 注意：验证/评估函数需在内部做好 label masking；若没有，请改为与训练同样方式构造 labels —— #
    from utils.eval_utils import evaluate_and_log  # 你已有的函数
    unpruned_ppl = evaluate_and_log(
        model=model,
        model_name=config["model_name"],
        epoch=None,
        config={**config, 'sparsity': 1.0},
        device=accelerator.device,
        accelerator=accelerator
    )
    perplexity_history['unpruned_baseline'] = unpruned_ppl

    # 进入迭代
    current_sparsity = 0.0
    iteration = 0
    overall_step = 0

    while current_sparsity < target_sparsity * 100 and iteration < max_iterations:
        iteration += 1
        accelerator.print("\n" + "="*70)
        accelerator.print(f"🔄 ITERATION {iteration}")
        accelerator.print("="*70)

        # 1) 剪枝
        accelerator.print(f"\n✂️ Pruning {prune_percentage*100:.1f}% of current non-zero weights...")
        num_pruned = prune_percentage_of_nonzeros(model, percentage=prune_percentage)
        apply_mask(model)               # 确保立即生效
        freeze_non_pruned_params(model) # 只训被剪枝的线性层
        current_sparsity = calculate_sparsity(model)
        accelerator.print(f"  📊 Current Sparsity: {current_sparsity:.2f}%")
        accelerator.print(f"  🎯 Target Sparsity: {target_sparsity*100:.1f}%")
        accelerator.print(f"  ✅ Pruned {num_pruned} parameter groups")

        # 剪枝后 PPL
        post_prune_ppl = evaluate_and_log(
            model=model,
            model_name=config["model_name"],
            epoch=None,
            config={**config, 'sparsity': 1.0 - current_sparsity/100},
            device=accelerator.device,
            accelerator=accelerator
        )

        # 2) 训练 epochs_per_iteration 轮（每轮都重新 build opt/scheduler，且仅包含可训练参数）
        # 动态训练：至少训练 epochs_per_iteration 个 epoch，如果 PPL 下降幅度 > 2 则继续训练
        accelerator.print(f"\n📚 Training for at least {epochs_per_iteration} epochs (with dynamic extension)...")
        
        # 计算总的训练步数（预估最大值，用于 scheduler）
        max_total_epochs = epochs_per_iteration + 20  # 预留额外的 epoch 空间
        optimizer = torch.optim.AdamW(get_trainable_params(model), lr=lr_sparse, weight_decay=2e-4)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2,
            num_training_steps=(config["max_steps"] * max_total_epochs) // gradient_accumulation_steps,
        )
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

        base = unwrap(accelerator, model)
        PRUNED_NAMES = pruned_param_name_set(base)
        SNAP_BEFORE = snapshot_params(base)  # 变化监控（可选）

        iteration_losses = []
        epoch_perplexities = []  # 记录每个 epoch 的 perplexity
        ppl_improvement_threshold = config.get("ppl_improvement_threshold", 2.0)  # PPL 下降幅度阈值
        
        epoch = 0
        continue_training = True
        
        while continue_training:
            epoch += 1
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"📊 Iteration {iteration}, Epoch {epoch} (min: {epochs_per_iteration})")
            accelerator.print(f"{'='*70}")

            with TorchTracemalloc() as tracemalloc:
                model.train()
                epoch_loss, num_batches = 0.0, 0

                progress_bar = tqdm(
                    enumerate(train_dataloader),
                    total=config['max_steps'],
                    desc=f"Epoch {epoch}",
                    disable=not accelerator.is_local_main_process
                )

                for step, batch in progress_bar:
                    if step == config['max_steps']:
                        break

                    # Label masking：pad 处 label=-100，避免在 pad 上学习导致 PPL 爆炸
                    labels = batch['input_ids'].clone()
                    if 'attention_mask' in batch:
                        labels[batch['attention_mask'] == 0] = -100

                    outputs = model(
                        input_ids=batch['input_ids'].to(accelerator.device),
                        attention_mask=batch.get('attention_mask', None),
                        labels=labels.to(accelerator.device),
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                    epoch_loss += loss.detach().float().item()
                    num_batches += 1

                    accelerator.backward(loss)
                    # 🔒 梯度裁剪：稀疏阶段非常关键，防止爆梯度
                    accelerator.clip_grad_norm_(get_trainable_params(model), 1.0)

                    if step % gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        apply_mask(model)  # 防止被剪的权重复活

                    overall_step += 1

                    progress_bar.set_postfix({
                        'loss': f'{(epoch_loss/num_batches):.4f}',
                        'step': f'{step}/{config["max_steps"]}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })

            final_avg_loss = epoch_loss / max(1, num_batches)
            iteration_losses.append(final_avg_loss)
            
            # 每个 epoch 后评估 perplexity
            accelerator.print(f"\n🔍 Evaluating after Epoch {epoch}...")
            current_epoch_ppl = evaluate_and_log(
                model=model,
                model_name=config["model_name"],
                epoch=None,
                config={**config, 'sparsity': 1.0 - current_sparsity/100},
                device=accelerator.device,
                accelerator=accelerator
            )
            epoch_perplexities.append(current_epoch_ppl)
            
            accelerator.print(f"\n{'='*70}")
            accelerator.print(f"✅ Iteration {iteration}, Epoch {epoch} Complete")
            accelerator.print(f"{'='*70}")
            accelerator.print(f"  📊 Average Loss: {final_avg_loss:.4f}")
            accelerator.print(f"  📈 Perplexity: {current_epoch_ppl:.2f}")
            accelerator.print(f"  🔢 Total Steps: {num_batches}")
            accelerator.print(f"\n💾 Memory Stats:")
            accelerator.print(f"  Memory before entering the train: {b2mb(tracemalloc.begin)} MB")
            accelerator.print(f"  Memory consumed (end-begin): {tracemalloc.used} MB")
            accelerator.print(f"  Peak Memory consumed (max-begin): {tracemalloc.peaked} MB")
            accelerator.print(
                f"  Total Peak Memory (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)} MB"
            )
            
            # 判断是否继续训练
            if epoch < epochs_per_iteration:
                # 还没达到最小 epoch 数，继续训练
                accelerator.print(f"  ⏩ Continue training (minimum epochs not reached: {epoch}/{epochs_per_iteration})")
                continue_training = True
            else:
                # 已达到最小 epoch 数，检查 PPL 改善情况
                if len(epoch_perplexities) >= 2:
                    ppl_improvement = epoch_perplexities[-2] - epoch_perplexities[-1]
                    accelerator.print(f"  📉 PPL Improvement: {ppl_improvement:.2f}")
                    
                    if ppl_improvement > ppl_improvement_threshold:
                        accelerator.print(f"  ⏩ Continue training (PPL improvement {ppl_improvement:.2f} > {ppl_improvement_threshold})")
                        continue_training = True
                    else:
                        accelerator.print(f"  ⏸️ Stop training (PPL improvement {ppl_improvement:.2f} <= {ppl_improvement_threshold})")
                        continue_training = False
                else:
                    # 只有一个 epoch 的数据，停止
                    continue_training = False


        # 训练后变化监控 - 详细检查
        SNAP_AFTER = snapshot_params(base)
        changed_pruned, changed_nonpruned = diff_report(SNAP_BEFORE, SNAP_AFTER, PRUNED_NAMES, eps=1e-7)
        
        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"🔍 PARAMETER CHANGE VERIFICATION - Iteration {iteration}")
        accelerator.print(f"{'='*70}")
        
        # 统计被prune层的变化
        accelerator.print(f"\n✅ Pruned Layers (Should Change):")
        accelerator.print(f"  Total pruned parameters changed: {len(changed_pruned)}")
        if changed_pruned:
            accelerator.print(f"  Top 10 changes:")
            for i, (n, d) in enumerate(changed_pruned[:10], 1):
                accelerator.print(f"    {i}. {n}: maxΔ={d:.2e}")
        else:
            accelerator.print(f"  ⚠️ WARNING: No pruned parameters changed!")
        
        # 统计非prune层的变化
        accelerator.print(f"\n❌ Non-Pruned Layers (Should NOT Change):")
        accelerator.print(f"  Total non-pruned parameters changed: {len(changed_nonpruned)}")
        if changed_nonpruned:
            accelerator.print(f"  ⚠️ WARNING: Non-pruned layers were modified!")
            accelerator.print(f"  Top 20 unexpected changes:")
            for i, (n, d) in enumerate(changed_nonpruned[:20], 1):
                if isinstance(d, str):
                    accelerator.print(f"    {i}. {n}: {d}")
                else:
                    accelerator.print(f"    {i}. {n}: maxΔ={d:.2e}")
        else:
            accelerator.print(f"  ✅ GOOD: All non-pruned layers remain frozen!")
        
        # 详细统计各类参数
        accelerator.print(f"\n📊 Detailed Parameter Statistics:")
        total_params = len(SNAP_BEFORE)
        total_changed = len(changed_pruned) + len(changed_nonpruned)
        total_unchanged = total_params - total_changed
        
        accelerator.print(f"  Total parameters: {total_params}")
        accelerator.print(f"  Changed (pruned): {len(changed_pruned)}")
        accelerator.print(f"  Changed (non-pruned): {len(changed_nonpruned)}")
        accelerator.print(f"  Unchanged: {total_unchanged}")
        
        # 验证requires_grad设置
        accelerator.print(f"\n🔒 Gradient Settings Verification:")
        trainable_count = 0
        frozen_count = 0
        trainable_names = []
        
        for name, param in base.named_parameters():
            if param.requires_grad:
                trainable_count += 1
                trainable_names.append(name)
            else:
                frozen_count += 1
        
        accelerator.print(f"  Trainable parameters: {trainable_count}")
        accelerator.print(f"  Frozen parameters: {frozen_count}")
        
        if trainable_names:
            accelerator.print(f"  Trainable parameter list (first 20):")
            for i, name in enumerate(trainable_names[:20], 1):
                is_changed = any(name in changed_name for changed_name, _ in changed_pruned)
                status = "✅ Changed" if is_changed else "⚠️ No change"
                accelerator.print(f"    {i}. {name} - {status}")
        
        # 最终验证结果
        accelerator.print(f"\n{'='*70}")
        if len(changed_nonpruned) == 0 and len(changed_pruned) > 0:
            accelerator.print(f"✅ VERIFICATION PASSED: Training only affected pruned layers!")
        elif len(changed_nonpruned) > 0:
            accelerator.print(f"❌ VERIFICATION FAILED: {len(changed_nonpruned)} non-pruned layers were modified!")
        else:
            accelerator.print(f"⚠️ WARNING: No parameters changed during training!")
        accelerator.print(f"{'='*70}\n")


        # 3) 本次迭代后的最终 perplexity（使用最后一个 epoch 的结果）
        post_train_ppl = epoch_perplexities[-1] if epoch_perplexities else post_prune_ppl
        accelerator.print(f"\n🔍 Final Perplexity after Iteration {iteration}: {post_train_ppl:.2f}")


        # 保存验证信息
        verification_info = {
            'pruned_params_changed': len(changed_pruned),
            'non_pruned_params_changed': len(changed_nonpruned),
            'total_params': len(SNAP_BEFORE),
            'verification_passed': len(changed_nonpruned) == 0 and len(changed_pruned) > 0,
            'trainable_params_count': trainable_count,
            'frozen_params_count': frozen_count,
        }
        
        # 添加变化最大的参数信息
        if changed_pruned:
            verification_info['top_pruned_changes'] = [
                {'name': n, 'max_delta': float(d) if not isinstance(d, str) else d} 
                for n, d in changed_pruned[:10]
            ]
        
        if changed_nonpruned:
            verification_info['top_non_pruned_changes'] = [
                {'name': n, 'max_delta': float(d) if not isinstance(d, str) else d} 
                for n, d in changed_nonpruned[:10]
            ]
        
        perplexity_history['iterations'].append({
            'iteration': iteration,
            'sparsity': current_sparsity,
            'post_prune_ppl': post_prune_ppl,
            'post_train_ppl': post_train_ppl,
            'avg_loss': sum(iteration_losses) / max(1, len(iteration_losses)),
            'total_epochs': epoch,
            'epoch_losses': iteration_losses,
            'epoch_perplexities': epoch_perplexities,
            'verification': verification_info
        })



        accelerator.print(f"\n📊 Iteration {iteration} Summary:")
        accelerator.print(f"  ✂️ Sparsity: {current_sparsity:.2f}%")
        accelerator.print(f"  📉 After Pruning: {post_prune_ppl:.2f}")
        accelerator.print(f"  📈 After Training: {post_train_ppl:.2f}")
        recovery = ((post_prune_ppl - post_train_ppl) / max(1e-8, post_prune_ppl)) * 100
        accelerator.print(f"  🎯 Recovery: {recovery:.1f}%")
        
        # 每个 iteration 后保存一次历史记录（增量保存）
        if accelerator.is_main_process:
            os.makedirs('training_logs', exist_ok=True)
            history_file = f'training_logs/{config["model_name"]}-iterative-history.json'
            
            temp_history = {
                'unpruned_baseline': perplexity_history['unpruned_baseline'],
                'iterations': perplexity_history['iterations'],
                'config': {
                    'model_name': config['model_name'],
                    'target_sparsity': target_sparsity,
                    'prune_percentage': prune_percentage,
                    'epochs_per_iteration': epochs_per_iteration,
                    'ppl_improvement_threshold': config.get("ppl_improvement_threshold", 2.0),
                    'learning_rate': lr_sparse,
                    'batch_size': batch_size,
                    'max_steps': config['max_steps'],
                    'seed': seed,
                }
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(temp_history, f, indent=2, ensure_ascii=False)
            
            accelerator.print(f"  💾 Progress saved to {history_file}")

        if current_sparsity >= target_sparsity * 100:
            accelerator.print(f"\n🎉 Target sparsity {target_sparsity*100:.1f}% reached!")
            break


    # 终汇总
    accelerator.print("\n" + "="*70)
    accelerator.print("📊 ITERATIVE PRUNING COMPLETE")
    accelerator.print("="*70)
    accelerator.print(f"  🔢 Total Iterations: {iteration}")
    accelerator.print(f"  ✂️ Final Sparsity: {current_sparsity:.2f}%")
    accelerator.print(f"  🟢 Unpruned Baseline: {perplexity_history['unpruned_baseline']:.2f}")

    if perplexity_history['iterations']:
        accelerator.print(f"\n📈 Iteration History:")
        for it in perplexity_history['iterations']:
            accelerator.print(
                f"  Iteration {it['iteration']}: Sparsity={it['sparsity']:.1f}%, "
                f"PPL={it['post_train_ppl']:.2f}, Epochs={it['total_epochs']}"
            )
        final_ppl = perplexity_history['iterations'][-1]['post_train_ppl']
        baseline_ppl = perplexity_history['unpruned_baseline']
        degradation = ((final_ppl - baseline_ppl) / max(1e-8, baseline_ppl)) * 100
        accelerator.print(f"\n🎯 Final Results:")
        accelerator.print(f"  Final Perplexity: {final_ppl:.2f}")
        accelerator.print(f"  vs Unpruned: {degradation:+.1f}%")

    accelerator.print("="*70)
    
    # 保存训练历史到 JSON 文件
    if accelerator.is_main_process:
        os.makedirs('training_logs', exist_ok=True)
        history_file = f'training_logs/{config["model_name"]}-iterative-history.json'
        
        # 添加配置信息到历史记录
        perplexity_history['config'] = {
            'model_name': config['model_name'],
            'target_sparsity': target_sparsity,
            'prune_percentage': prune_percentage,
            'epochs_per_iteration': epochs_per_iteration,
            'ppl_improvement_threshold': config.get("ppl_improvement_threshold", 2.0),
            'learning_rate': lr_sparse,
            'batch_size': batch_size,
            'max_steps': config['max_steps'],
            'seed': seed,
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(perplexity_history, f, indent=2, ensure_ascii=False)
        
        accelerator.print(f"\n💾 Training history saved to {history_file}")


    # 保存
    if not config.get('save_model') or config['save_model']:
        os.makedirs('pruned_models', exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        final_sparsity = current_sparsity / 100
        torch.save(unwrapped.state_dict(), f'pruned_models/{config["model_name"]}-iterative-{final_sparsity:.2f}.pt')
        accelerator.print(f"\n✅ Model saved to pruned_models/{config['model_name']}-iterative-{final_sparsity:.2f}.pt")


# =============================
# Wrapper (CLI style)
# =============================
def iterative_training_wrapper(config):
    parser = argparse.ArgumentParser(description="Iterative Pruning Training")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--cpu", action="store_true", help="Train on CPU")
    parser.add_argument("--checkpointing_steps", type=str, default=None,
                        help="Checkpointing frequency")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--with_tracking", action="store_true",
                        help="Enable experiment tracking")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Logging directory")

    # 你原本就是直接传 ['--mixed_precision','fp16']，这里保持一致：
    args = parser.parse_args(['--mixed_precision', 'fp16'])
    iterative_training(config, args)
