# 极限压缩实验 - 功能增强说明

本文档说明了对极限压缩实验脚本的所有功能增强。

## 🎯 主要功能

### 1. **自动 Perplexity 评估**

训练过程中会自动在以下时间点评估 Perplexity：

- ✅ **训练前基线评估**：在第一个 epoch 开始前评估原始模型
- ✅ **每个 epoch 后评估**：每个训练 epoch 结束后立即评估
- ✅ **趋势跟踪**：实时显示相对基线的改进百分比
- ✅ **最终总结**：训练结束后显示最佳/最终 perplexity

### 2. **智能批大小管理**

根据批大小自动调整训练步数和学习率：

```python
# 示例：保持总数据量不变
batch_size=1  → max_step=10000, lr=5e-6
batch_size=4  → max_step=2500,  lr=1e-5
batch_size=8  → max_step=1250,  lr=1.41e-5
```

**公式**：
- `max_step = 10000 / batch_size`
- `lr = base_lr * sqrt(batch_size)`

### 3. **详细训练日志**

每个 epoch 显示：
- 📊 平均 Loss
- 🔢 处理的批次数
- 💾 GPU 显存使用统计
- 📈 Perplexity 变化趋势

### 4. **命令行参数**

```powershell
# 基础用法
python .\extreme_compression_experiment.py --sparsity 0.1

# 指定批大小（加速训练）
python .\extreme_compression_experiment.py --sparsity 0.1 --batch-size 4

# 完整示例
python .\extreme_compression_experiment.py \
    --sparsity 0.1 \
    --model opt-350m \
    --batch-size 8
```

## 📊 输出示例

### 训练前基线评估

```
======================================================================
📊 Baseline Evaluation (Before Training)
======================================================================
  📖 Loading Wikitext-2 test set...
  📏 Sequence length: 245566
  🧮 Computing perplexity: 100%|████████████| 120/120
  ✅ Baseline Perplexity: 27.45
======================================================================
```

### Epoch 结束评估

```
======================================================================
✅ Epoch 1 Complete
======================================================================
  📊 Average Loss: 3.2145
  🔢 Batches Processed: 10000

💾 Memory Stats:
  Memory before entering the train: 478 MB
  Memory consumed (end-begin): 1279 MB
  Peak Memory consumed (max-begin): 2260 MB
  Total Peak Memory (max): 2738 MB

🔍 Evaluating model after Epoch 1...
======================================================================
📊 Evaluation After Epoch 1
======================================================================
  📖 Loading Wikitext-2 test set...
  🧮 Computing perplexity: 100%|████████████| 120/120
  ✅ Perplexity: 25.82
  📈 Epoch: 1/5
  🎯 Sparsity: 0.1
======================================================================

📈 Perplexity Trend:
  Baseline: 27.45
  Epoch 1: 25.82 (↓5.9%)
```

### 最终总结

```
======================================================================
📊 Final Perplexity Summary
======================================================================
  Baseline: 27.45
  Final (Epoch 5): 23.67
  Best (Epoch 4): 23.51
  Total Improvement: +13.8%
======================================================================
```

## 🔧 技术细节

### Perplexity 计算

基于 `Testing.ipynb` 的实现：

```python
def evaluate_perplexity(model, model_name, device='cuda', 
                       token_length=2048, stride=2048):
    """
    使用滑动窗口在 Wikitext-2 测试集上计算 perplexity
    
    - Token length: 2048
    - Stride: 2048 (无重叠)
    - Dataset: Wikitext-2-raw-v1 test split
    """
```

### 评估时机

1. **训练前**：
   ```python
   baseline_ppl = evaluate_and_log(
       model=model,
       epoch=None,  # None = baseline
       ...
   )
   ```

2. **每个 epoch 后**：
   ```python
   for epoch in range(num_epochs):
       # ... training ...
       
       epoch_ppl = evaluate_and_log(
           model=model,
           epoch=epoch,
           ...
       )
   ```

### 显存管理

- 评估时自动切换到 `model.eval()` 模式
- 使用 `torch.no_grad()` 减少显存占用
- 评估后自动切回 `model.train()` 模式

## 📈 性能对比

### 不同批大小的训练时间（估算）

| Batch Size | Steps/Epoch | 训练时间 | 评估时间 | 总时间 | 显存需求 |
|-----------|------------|---------|---------|--------|---------|
| 1 | 10000 | 50min | 3min | ~53min | ~2GB |
| 2 | 5000 | 30min | 3min | ~33min | ~3GB |
| 4 | 2500 | 18min | 3min | ~21min | ~5GB |
| 8 | 1250 | 12min | 3min | ~15min | ~8GB |

> 注意：评估时间对所有配置相同，因为评估不受训练批大小影响

## 🎓 使用建议

### 1. 低显存场景（≤4GB）
```powershell
python .\extreme_compression_experiment.py --sparsity 0.1 --batch-size 1
```

### 2. 中等显存场景（6-8GB）
```powershell
python .\extreme_compression_experiment.py --sparsity 0.1 --batch-size 4
```

### 3. 高显存场景（≥12GB）
```powershell
python .\extreme_compression_experiment.py --sparsity 0.1 --batch-size 8
```

### 4. 快速测试（减少步数）

编辑 `extreme_compression_experiment.py`：
```python
base_samples = 1000  # 从 10000 改为 1000
config["num_epochs"] = 2  # 从 5 改为 2
```

## 📝 实验日志

所有 perplexity 结果会在训练过程中实时显示，建议：

1. **重定向到文件**：
   ```powershell
   python .\extreme_compression_experiment.py --sparsity 0.1 2>&1 | Tee-Object -FilePath training_log.txt
   ```

2. **使用 WandB 跟踪**（需要在 `fsdp_finetune.py` 中启用 `--with_tracking`）

## 🐛 故障排除

### 问题：评估时显存不足

**解决方案**：减小 `token_length`

编辑 `utils/eval_utils.py`：
```python
def evaluate_perplexity(..., token_length=1024, stride=1024):  # 从 2048 改为 1024
```

### 问题：评估时间过长

**解决方案**：增大 `stride`

```python
def evaluate_perplexity(..., stride=4096):  # 从 2048 改为 4096
```
这会减少计算窗口数量，但可能略微影响精度。

### 问题：Baseline perplexity 异常高

**可能原因**：
1. 剪枝模型加载失败
2. 稀疏度设置过高（如 0.05）
3. 模型架构不匹配

**检查方法**：
```python
# 在评估前打印非零参数比例
for name, param in model.named_parameters():
    if 'weight' in name:
        nonzero_ratio = (param != 0).float().mean()
        print(f"{name}: {nonzero_ratio:.2%} non-zero")
```

## 📚 相关文件

- `extreme_compression_experiment.py` - 主实验脚本
- `fsdp_finetune.py` - FSDP 训练核心逻辑
- `utils/eval_utils.py` - Perplexity 评估工具
- `Testing.ipynb` - 原始评估参考实现

## 🔄 未来改进

- [ ] 支持更多评估指标（BLEU, ROUGE 等）
- [ ] 添加早停（Early Stopping）基于 perplexity
- [ ] 支持自定义评估数据集
- [ ] 添加 TensorBoard 可视化
- [ ] 自动保存最佳 perplexity checkpoint

---

**最后更新**：2025年10月21日
