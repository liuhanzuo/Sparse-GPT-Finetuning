#!/usr/bin/env python
"""
Iterative Pruning Training Script

This script implements iterative magnitude pruning:
- Start with an unpruned model
- Iteratively prune 10% of non-zero weights
- Train for 3 epochs after each pruning step
- Continue until target sparsity is reached

Example usage:
    python run_iterative_train.py --model opt-125m --target-sparsity 0.9 --batch-size 8
"""

import argparse
from iterative_train import iterative_training_wrapper


def main():
    parser = argparse.ArgumentParser(description="Iterative Pruning Training")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="opt-125m",
                       choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b"],
                       help="Model to train")
    
    # Pruning configuration
    parser.add_argument("--target-sparsity", type=float, default=0.9,
                       help="Target sparsity (e.g., 0.9 = 90%% zeros)")
    parser.add_argument("--prune-percentage", type=float, default=0.1,
                       help="Percentage of non-zeros to prune each iteration")
    parser.add_argument("--epochs-per-iteration", type=int, default=3,
                       help="Number of epochs to train after each pruning")
    parser.add_argument("--max-iterations", type=int, default=20,
                       help="Maximum number of pruning iterations")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=1250,
                       help="Maximum training steps per epoch")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of epochs to train after each pruning")
    parser.add_argument("--ppl-improvement-threshold", type=float, default=2.0,
                       help="PPL improvement threshold for continuing training")

    
    # System configuration  
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-model", action="store_true", default=True,
                       help="Save the final model")
    
    args = parser.parse_args()
    
    # Build config
    config = {
        "model_name": args.model,
        "target_sparsity": args.target_sparsity,
        "prune_percentage": args.prune_percentage,
        "epochs_per_iteration": args.epochs_per_iteration,
        "max_iterations": args.max_iterations,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "save_model": args.save_model,
        "ppl_improvement_threshold": args.ppl_improvement_threshold,
    }

    
    print("="*70)
    print("🔄 ITERATIVE PRUNING CONFIGURATION")
    print("="*70)
    print(f"  Model: {config['model_name']}")
    print(f"  Target Sparsity: {config['target_sparsity']*100:.1f}%")
    print(f"  Prune per iteration: {config['prune_percentage']*100:.1f}% of non-zeros")
    print(f"  Epochs per iteration (minimum): {config['epochs_per_iteration']}")
    print(f"  PPL improvement threshold: {config['ppl_improvement_threshold']}")
    print(f"  Max iterations: {config['max_iterations']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Max steps per epoch: {config['max_steps']}")
    print("="*70)

    
    # Run training
    iterative_training_wrapper(config)
    
    print("\n🎉 Iterative pruning training complete!")


if __name__ == "__main__":
    main()
