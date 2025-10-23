"""
极限压缩实验：90%+ 参数剪枝
警告：这是一个高风险实验，模型性能会大幅下降

使用方法：
1. 确保已经运行过 run_fsdp_finetune.py 验证环境
2. (可选) 先用 SparseGPT.ipynb 生成 pruned_models/opt-125m-0.1.pt
3. 运行本脚本进行极限压缩微调

注意：
- 如果已有 0.1 稀疏度的剪枝模型，将加载它
- 如果没有，将直接在未剪枝模型上训练（效果会差很多）
- 推荐先用 SparseGPT.ipynb 生成剪枝模型
"""

from fsdp_finetune import fsdp_finetune
import os

def extreme_compression_experiment(target_sparsity=0.1, model_name="opt-125m", batch_size=1):
    """
    极限压缩实验
    
    Args:
        target_sparsity: 目标稀疏度，0.1 = 90% 压缩，0.05 = 95% 压缩
        model_name: 模型名称
        batch_size: 批大小，增加可以加速训练但需要更多显存
    """
    
    compression_rate = (1 - target_sparsity) * 100
    remaining_params = target_sparsity * 100
    
    print("=" * 70)
    print("🚀 极限压缩实验：SparseGPT 高稀疏度微调")
    print("=" * 70)
    print(f"模型: facebook/{model_name}")
    print(f"目标稀疏度: {target_sparsity}")
    print(f"保留参数: {remaining_params:.1f}%")
    print(f"压缩率: {compression_rate:.1f}%")
    print("=" * 70)
    
    # 检查是否存在剪枝模型
    pruned_model_path = f'pruned_models/{model_name}-{target_sparsity}.pt'
    has_pruned_model = os.path.exists(pruned_model_path)
    
    if has_pruned_model:
        print(f"✅ 找到剪枝模型: {pruned_model_path}")
        print("   将加载并微调剪枝后的模型")
    else:
        print(f"⚠️  未找到剪枝模型: {pruned_model_path}")
        print("   将使用未剪枝的模型训练（效果会差很多）")
        print()
        print("🔧 建议步骤：")
        print("   1. 打开 SparseGPT.ipynb")
        print(f"   2. 设置 SPARSENESS_LIST = [{target_sparsity}]")
        print("   3. 运行 notebook 生成剪枝模型")
        print("   4. 再运行本脚本进行微调")
        print()
        response = input("是否继续使用未剪枝模型? (y/N): ")
        if response.lower() != 'y':
            print("实验取消")
            return
    
    print("=" * 70)
    
    # 根据 batch_size 自动调整 max_step
    # 保持总数据量约 10,000 个样本
    base_samples = 10000
    max_step = base_samples // batch_size
    
    # 根据 batch_size 调整学习率
    # 经验法则：batch_size 翻倍，学习率也可以适当增加（但不要线性增加）
    base_lr = 5e-6
    lr = base_lr * (batch_size ** 0.5)  # 使用平方根缩放
    
    # 配置参数
    config = {
        "lr": lr,
        "num_epochs": 10,
        "seed": 42,
        "batch_size": batch_size,
        "model_name": model_name,
        "sparsity": target_sparsity,
        "train_steps": max_step,
        "max_step": max_step,
        "save_model": True,
        "resume":True,
        "use_lora_per_layer": True,
        "lora_epochs": 5,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "unfreeze_base": True
    }
    
    print("📋 训练配置:")
    print(f"   批大小: {config['batch_size']}")
    print(f"   学习率: {config['lr']:.2e} (自动调整)")
    print(f"   训练轮次: {config['num_epochs']}")
    print(f"   每轮最大步数: {config['max_step']}")
    print(f"   总样本数/轮: ~{batch_size * max_step}")
    print(f"   预期优化器更新/轮: ~{max_step}")
    print("=" * 70)
    print("⚠️  警告：")
    print(f"   - {compression_rate:.0f}% 压缩率会导致显著性能下降")
    if batch_size > 1:
        print(f"   - Batch size={batch_size} 需要 ~{batch_size}GB 显存")
    print("   - 训练时间较长，预计 30-60 分钟")
    print("   - 建议从 0.2 稀疏度开始逐步尝试")
    print("=" * 70)
    
    input("按 Enter 开始实验...")
    
    print("\n🏃 开始训练...\n")
    
    try:
        fsdp_finetune(config)
        
        print("\n" + "=" * 70)
        print("✅ 实验完成！")
        print(f"📦 模型已保存到: pruned_models/{model_name}-{target_sparsity}-finetuned.pt")
        print("=" * 70)
        print("\n📊 后续步骤：")
        print("   1. 使用 Testing.ipynb 评估模型性能 (Perplexity)")
        print("   2. 测试生成质量")
        print("   3. 与其他稀疏度的模型对比")
        print()
        print("💡 提示：")
        print("   - 如果效果不理想，尝试增加 num_epochs 或调整 batch_size")
        print("   - 可以先用 0.2 或 0.15 稀疏度验证流程")
        print("   - 更大的模型 (opt-350m, opt-1.3b) 可能对极限压缩更鲁棒")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ 实验失败！")
        print(f"错误信息: {str(e)}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="极限压缩实验")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="目标稀疏度 (0.1=90%%压缩, 0.05=95%%压缩)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opt-125m",
        choices=["opt-125m", "opt-350m", "opt-1.3b"],
        help="模型名称"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批大小 (1=省显存, 4=更快)"
    )
    
    args = parser.parse_args()
    
    extreme_compression_experiment(
        target_sparsity=args.sparsity,
        model_name=args.model,
        batch_size=args.batch_size
    )