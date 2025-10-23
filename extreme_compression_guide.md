# 极限压缩指南：90%+ 压缩率实验

本指南介绍如何使用 SparseGPT 实现 **90% 以上的极限压缩率**（sparsity ≤ 0.1，仅保留 ≤10% 参数）。

## 📊 实验证据

仓库中的 `Testing.ipynb` 已经测试过 **0.1 稀疏度（90% 压缩）**，证明极限压缩是**技术上可行的**。

```python
# 来自 Testing.ipynb line 156
SPARSITIES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
```

## ⚠️ 关键警告

### 压缩率 vs 模型性能
| 稀疏度 | 保留参数 | 剪枝比例 | 难度等级 | 预期效果 |
|--------|----------|----------|----------|----------|
| 0.5 | 50% | 50% | ⭐ 简单 | 性能下降小，推荐起点 |
| 0.3 | 30% | 70% | ⭐⭐ 中等 | 明显下降，需要微调 |
| 0.2 | 20% | 80% | ⭐⭐⭐ 困难 | 显著下降，需精细调参 |
| **0.1** | **10%** | **90%** | ⭐⭐⭐⭐ 极难 | **极限压缩，性能严重受限** |
| 0.05 | 5% | 95% | ⭐⭐⭐⭐⭐ 极限 | 实验性，可能几乎不可用 |

### 重要提示
1. **0.1 稀疏度是极限**：性能会大幅下降，但仍可能保持基本生成能力
2. **必须使用迭代剪枝+微调**：一次性剪枝到 90% 几乎必然失败
3. **需要更多校准数据**：建议增加 `calibration_size` 到 256 或更多
4. **需要更长微调时间**：建议 `num_epochs >= 5`

## 🚀 实现方案

### 方案 A：逐步压缩（推荐）

**最安全的方式**：从低压缩率逐步增加到 90%+

```python
# 在 SparseGPT.ipynb 或 Iterative_Pruning.ipynb 中配置

# 第一轮：温和压缩
SPARSENESS_LIST = [0.7, 0.5, 0.3]
calibration_size = 256  # 增加到 256
num_epochs = 3

# 第二轮：高压缩（基于第一轮最佳结果）
SPARSENESS_LIST = [0.2, 0.15, 0.1]
calibration_size = 512  # 进一步增加
num_epochs = 5

# 第三轮：极限压缩（可选，高风险）
SPARSENESS_LIST = [0.08, 0.05]
calibration_size = 1024
num_epochs = 10
```

### 方案 B：直接极限压缩（高风险）

如果你有充足的 GPU 资源和时间，可以直接尝试：

```python
# 在 run_fsdp_finetune.py 中配置
config = {
    "lr": 1e-5,              # 降低学习率避免震荡
    "num_epochs": 10,        # 增加训练轮次
    "seed": 42,
    "batch_size": 1,         # 显存有限时保持为 1
    "model_name": "opt-125m",  # 从小模型开始
    "sparsity": 0.1,         # 90% 压缩！
    "train_steps": 5000,     # 大幅增加训练步数
    "max_step": 5000,
    "save_model": True,
}
```

### 方案 C：使用 Testing.ipynb 复现原始实验

```python
# 在 Testing.ipynb 中取消注释已有的 0.1 配置
SPARSITIES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]  # line 156

# 调整校准参数以获得更好结果
CALIBRATION_SIZE = 512       # 增加校准数据量
TOKEN_LENGTH = 1024          # 增加 token 长度
CALIBRATION_BATCH_SIZE = 2   # 适当增加批大小

# SparseGPT 参数（保持默认或微调）
EPSILON = 1e-8
B = 128
Bs = 128
```

## 📝 完整实验流程（90% 压缩）

### Step 1: 准备环境
```powershell
# 确保环境已就绪
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: 修改 SparseGPT.ipynb 进行极限剪枝

打开 `SparseGPT.ipynb`，定位到参数配置区域并修改：

```python
# 找到这些行并修改为：
model_size = "opt-125m"           # 从小模型开始
calibration_size = 512            # 增加校准数据
token_length = 1024               # 增加 token 长度
calibration_batch_size = 2        # 适当批大小
EPSILON = 1e-8
B = 128
Bs = 128

# 关键：设置极限稀疏度
SPARSENESS_LIST = [0.1]  # 90% 压缩！
# 或者保守一点：
# SPARSENESS_LIST = [0.15, 0.1]  # 85% 和 90%
```

运行 notebook，生成 `pruned_models/opt-125m-0.1.pt`

### Step 3: 使用 Iterative_Pruning.ipynb 进行迭代微调

```python
model_size in ['opt-125m']
SPARSENESS_LIST = [0.1]  # 对应刚才生成的剪枝模型

# 增加微调强度
num_epochs = 10  # 更多训练轮次
train_steps = 5000  # 更多步数
```

### Step 4: 评估模型性能

使用 `Testing.ipynb` 或 `Paper_Results.ipynb` 中的评估代码：

```python
# 加载你的 90% 压缩模型
from utils.save_utils import load_masked_model_single

model = OPTForCausalLM.from_pretrained('facebook/opt-125m')
load_masked_model_single(model, 'pruned_models/opt-125m-0.1-finetuned.pt')

# 在 Wikitext 上评估 Perplexity
# （参考 Testing.ipynb 中的评估代码）
```

## 🎯 关键优化技巧

### 1. 增加校准数据质量
```python
# 使用更高质量的校准数据
calibration_size = 512   # 原始 128 → 512
token_length = 2048      # 原始 512 → 2048（如果显存允许）
```

### 2. 调整 SparseGPT 超参数
```python
# 更细粒度的块大小
B = 256    # 原始 128 → 256
Bs = 256   # 原始 128 → 256

# 更保守的数值稳定性
EPSILON = 1e-10  # 原始 1e-8 → 1e-10
```

### 3. 更激进的微调策略
```python
# 更低的学习率
lr = 5e-6  # 原始 2e-5 → 5e-6

# 更多训练步数
num_epochs = 20
train_steps = 10000
max_step = 10000
```

### 4. 逐层分析（调试用）
```python
# 查看每层的稀疏度分布
for name, param in model.named_parameters():
    if 'weight' in name:
        sparsity = (param == 0).float().mean()
        print(f"{name}: {sparsity:.2%} zeros")
```

## 📊 预期结果

### 性能指标参考（OPT-125M on Wikitext）

根据 README 描述和代码注释推测：

| 稀疏度 | Perplexity (估计) | 相对基线 | 可用性 |
|--------|-------------------|----------|--------|
| 1.0 (基线) | ~27 | 100% | ✅ 完全正常 |
| 0.5 | ~29-31 | 93-96% | ✅ 轻微下降 |
| 0.3 | ~34-38 | 71-79% | ⚠️ 明显下降 |
| 0.2 | ~42-50 | 54-64% | ⚠️ 显著下降 |
| **0.1** | **~60-100+** | **<45%** | ⚠️ **严重下降，仍可生成** |
| 0.05 | ~150+ | <18% | ❌ 几乎不可用 |

> 注意：以上数字为推测，实际效果取决于具体实现和微调质量

## 🔬 实验建议

### 保守策略（推荐新手）
```python
# Step 1: 先达到 80% 压缩
SPARSENESS_LIST = [0.5, 0.3, 0.2]

# Step 2: 评估 0.2 的效果，如果可接受再尝试 0.1
```

### 激进策略（有经验）
```python
# 直接挑战 90%+
SPARSENESS_LIST = [0.2, 0.15, 0.1, 0.08]

# 准备大量计算资源和时间
calibration_size = 1024
num_epochs = 15-20
```

### 极限策略（研究用途）
```python
# 探索理论极限
SPARSENESS_LIST = [0.1, 0.08, 0.05, 0.03, 0.01]

# 每个稀疏度都需要大量微调
num_epochs = 30+
train_steps = 50000+
```

## ⚡ 快速开始：一键 90% 压缩实验

创建一个新的 Python 脚本 `extreme_compression_experiment.py`：

```python
"""
极限压缩实验：90% 参数剪枝
警告：这是一个高风险实验，模型性能会大幅下降
"""

from fsdp_finetune import fsdp_finetune

# 配置：90% 压缩（sparsity = 0.1）
config = {
    "lr": 5e-6,              # 降低学习率
    "num_epochs": 10,        # 增加训练轮次
    "seed": 42,
    "batch_size": 1,         # 显存有限保持为 1
    "model_name": "opt-125m",  # 从小模型开始
    "sparsity": 0.1,         # 🎯 90% 压缩！
    "train_steps": 10000,    # 大量训练步数
    "max_step": 10000,
    "save_model": True,
}

print("=" * 60)
print("🚀 极限压缩实验：90% 参数剪枝")
print("=" * 60)
print(f"模型: {config['model_name']}")
print(f"稀疏度: {config['sparsity']} (保留 {config['sparsity']*100:.0f}% 参数)")
print(f"压缩率: {(1-config['sparsity'])*100:.0f}%")
print(f"训练轮次: {config['num_epochs']}")
print(f"总步数: {config['max_step']}")
print("=" * 60)
print("⚠️  警告：极限压缩会导致显著性能下降")
print("⏱️  预计运行时间：30-60 分钟（取决于硬件）")
print("=" * 60)

input("按 Enter 继续实验...")

fsdp_finetune(config)

print("\n" + "=" * 60)
print("✅ 实验完成！")
print(f"模型已保存到: pruned_models/opt-125m-0.1-finetuned.pt")
print("=" * 60)
```

运行：
```powershell
python extreme_compression_experiment.py
```

## 🎓 理论极限

根据神经网络剪枝研究和 SparseGPT 论文：

- **90% 压缩 (0.1 稀疏度)**：理论上可行，但需要精细调参
- **95% 压缩 (0.05 稀疏度)**：极限挑战，性能严重受限
- **98%+ 压缩 (< 0.02 稀疏度)**：基本不可行，除非使用特殊技术（如结构化剪枝、知识蒸馏等）

## 📚 参考资源

1. **SparseGPT 原始论文**：查找关于极限稀疏度的实验结果
2. **仓库中的 Testing.ipynb**：已包含 0.1 稀疏度的测试代码
3. **Paper_Results.ipynb**：查看完整实验流程

## 🤝 需要帮助？

如果实验遇到问题：
1. 从 0.2 稀疏度开始，逐步降低到 0.1
2. 增加 `calibration_size` 和 `num_epochs`
3. 检查每层的稀疏度分布，避免某些层过度剪枝
4. 考虑使用更大的模型（OPT-350M, OPT-1.3B）可能对极限压缩更鲁棒

---

**最后提醒**：90%+ 压缩是一个研究级别的挑战，不要期待产品级的性能。建议：
- 学术研究：值得尝试，可以发表论文
- 实际应用：建议保持在 70-80% 压缩率（0.2-0.3 稀疏度）以保持可用性
