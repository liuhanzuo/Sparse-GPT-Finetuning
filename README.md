# Authors
- Aaquib Syed | [asyed04@umd.edu](mailto:asyed04@terpmail.umd.edu)
- Phillip Huang Guo | [phguo@umd.edu](mailto:phguo@terpmail.umd.edu)
- Vijaykaarti Sundarapandiyan | [vsundar1@umd.edu](mailto:vsundar1@terpmail.umd.edu)

# Sparse-GPT-Finetuning

Massive language models with billions of parameters have significant compute expenses and thus can benefit from pruning. 
Pruning techniques for massive models are typically iterative and require extensive weight retraining after pruning. 
SparseGPT, a recently introduced one-shot technique for pruning such models, enables pruning without retraining. 
We improve upon SparseGPT by fine-tuning during pruning with minimal training steps, and we perform experiments against magnitude 
pruning and find that our iteratively fine-tuned SparseGPT models significantly outperform their magnitude pruning counterparts at high sparsity.

# Usage

## Pruning and Tuning

SparseGPT.ipynb has code to prune and Finetuning.ipynb has code to finetune the pruned models. 
Use Iterative_Pruning.ipynb to iteratively prune and finetune using FullyShardedDataParallel.

### Cerebras/SparseGPT Pruning and FullyShardedDataParallel Tuning
#### Using Cerebras for magnitude pruning:
- Change the model name: ```model_name = "facebook/opt-125m"```
- Run the notebook

#### Using SparseGPT.ipynb for pruning:
- Change model size in ```model_size = "opt-125m"```
- Adjust following parameters:
  - Amount of sentences used for calibration: ```calibration_size=128```
  - Max length of tokens in a sentence: ```token_length=512```
  - Amount of batches for calibration: ```calibration_batch_size=2```
  - Small constant to add for matrix inverses: ```EPSILON = 1e-8```
  - Block size for pruning: ```B = 4```
  - Adaptive mask selection blocksize: ```Bs = 2```
- Adjust how many sparsities to generate: ```SPARSENESS_LIST = [0.5]```
- Run the notebook

#### Finetuning after pruning:
- Adjust model sizes to tune: ```model_size in ['opt-1.3b']```
- Adjust sparsities to tune: ```SPARSITIES = [1, 0.9, 0.7, 0.5, 0.3, 0.2]```
- Run the notebook

### Iterative Pruning and Tuning
- Change model sizes to prune ```model_size in ['opt-125m', 'opt-350m', 'opt-1.3b']```
- Adjust following parameters:
  - Amount of sentences used for calibration: ```calibration_size=128```
  - Max length of tokens in a sentence: ```token_length=512```
  - Amount of batches for calibration: ```calibration_batch_size=2```
  - Small constant to add for matrix inverses: ```EPSILON = 1e-8```
  - Block size for pruning: ```B = 4```
  - Adaptive mask selection blocksize: ```Bs = 2```
- Adjust how many sparsities to generate: ```SPARSENESS_LIST = [0.5]```
- Run the notebook

# Results
![Wikitext-OPT-125m-NonLog](https://user-images.githubusercontent.com/47124521/229620085-d6e3d4be-19e8-4c7a-be8e-8e6179af4ccd.png)
![Wikitext-OPT-125m-Log](https://user-images.githubusercontent.com/47124521/229620135-f0d9a44c-3737-4a6d-99e6-fb49594ea2f6.png)


![Wikitext-OPT-1 3b-NonLog](https://user-images.githubusercontent.com/47124521/229619883-9414a251-9915-45d7-baeb-79fc11dc05dc.png)
![Wikitext-OPT-1 3b-Log](https://user-images.githubusercontent.com/47124521/229619910-344a3783-e18d-479a-80b5-1073f1fbbdf8.png)

As the graphs in Figure 1 demonstrate, SparseGPT iterative pruning and fine-tuning is stronger than
every other technique beyond 0.4 sparseness on OPT-125M and 0.6 sparseness on OPT-1.3B. We
find that SparseGPT non-iterative pruning and fine-tuning is moderately successful compared to
no fine-tuning in all cases, but is beaten out significantly by both iterative pruning and fine-tuning
methods beyond 0.5 sparseness.

## Quickstart (Windows PowerShell)

The scripts and notebooks rely on PyTorch, Transformers, Datasets, and Accelerate. A minimal
way to smoke-test the training loop without modifying notebooks is to run a tiny FSDP
finetuning step on OPT-125M. The commands below assume a working CUDA setup. If you don't
have a GPU, you can still run but it will be slow and memory-limited.

1) Create and activate a virtual environment

```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```
pip install -r requirements.txt
pip install accelerate
```

If you need a specific CUDA-enabled torch, refer to https://pytorch.org/get-started/locally/
and install the matching wheel, for example:

```
# Optional: replace with versions matching your CUDA
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Run a tiny finetune smoke test (downloads facebook/opt-125m on first run)

```
python .\run_fsdp_finetune.py
```

This uses Wikitext-2 streaming with very small steps. It will finish quickly and not save a
checkpoint by default. To perform real finetuning:

- Edit `run_fsdp_finetune.py` and raise `num_epochs`, `batch_size`, `train_steps`, and `max_step`.
- Set `sparsity` to the value of an existing pruned checkpoint you created (e.g., `0.5`) and place
  it under `pruned_models/{model_name}-{sparsity}.pt`.
- For multi-GPU or advanced configs, consider using Hugging Face Accelerate config via
  `accelerate config` then adapting `fsdp_finetune.py` accordingly.

4) Notebooks

- `SparseGPT.ipynb`: prune models (one-shot SparseGPT)
- `Finetuning.ipynb`: finetune pruned models
- `Iterative_Pruning.ipynb`: iterative prune + finetune with FSDP

Tip: Large models (e.g., OPT-1.3B) require significant GPU memory. Start with OPT-125M, verify
the pipeline, then scale up.
