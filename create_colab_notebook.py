#!/usr/bin/env python3
"""
Script to create the vec2vec Colab notebook with VSP variants.
"""

import nbformat as nbf
import json

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Title and Introduction
    cells.append(nbf.v4.new_markdown_cell('''# vec2vec: Colab Reproduction with VSP Variants (~50k Examples)

This notebook reproduces the core vec2vec model from the paper ["Harnessing the Universal Geometry of Embeddings"](https://arxiv.org/abs/2505.12540) with modified VSP (Vector Space Preservation) variants.

## What This Notebook Does

We train vec2vec on **stella → gte** embedding translation using the **NQ dataset** with ~50,000 training examples, comparing three VSP loss variants:

1. **Original VSP** - Preserves pairwise dot-product similarities (as in the paper)
2. **Conformal VSP** - Preserves angles (cosine similarities) between vectors
3. **Topological VSP (kNN)** - Preserves k-nearest neighbor structure

## Goal

Explore whether the "universal geometry" of embeddings is:
- **Metric** (dot-product preservation - original)
- **Conformal** (angle preservation)
- **Topological** (neighborhood preservation)

## Metrics

For each variant, we evaluate:
- **Cosine Similarity** - Alignment between translated and ground-truth embeddings
- **Top-1 Accuracy** - Nearest neighbor retrieval accuracy
- **Mean Rank** - Average rank of correct target in similarity-ranked list
'''))

    # Cell 2: Environment Check
    cells.append(nbf.v4.new_code_cell('''# Check GPU availability and system info
!nvidia-smi

import sys
print(f"\\nPython version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("PyTorch not yet installed")
'''))

    # Cell 3: Clone Repository
    cells.append(nbf.v4.new_code_cell('''# Clone the vec2vec repository
import os

REPO_URL = "https://github.com/danielyxu/vec2vec.git"
REPO_DIR = "/content/vec2vec"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    print(f"Repository already exists at {REPO_DIR}")

%cd {REPO_DIR}
!git pull origin main
print(f"\\nWorking directory: {os.getcwd()}")
'''))

    # Cell 4: Install Dependencies
    cells.append(nbf.v4.new_code_cell('''# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers datasets accelerate wandb safetensors toml sentence-transformers
!pip install -q pandas matplotlib seaborn tqdm

# Verify installations
import torch
import transformers
import datasets
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
'''))

    # Cell 5: Data Preparation Explanation
    cells.append(nbf.v4.new_markdown_cell('''## Data & Embedding Preparation

The vec2vec framework uses **streaming embeddings** from pre-encoded text datasets. For this notebook:

- **Dataset**: Natural Questions (NQ) - a question-answering dataset
- **Source Embedding**: `stella` (infgrad/stella-base-en-v2) - 768 dimensions
- **Target Embedding**: `gte` (thenlper/gte-base) - 768 dimensions

We'll use:
- **~50,000 training examples** (subset of full dataset)
- **4,096 validation examples**
- **8,192 test examples** for final evaluation

The data loading happens on-the-fly during training using HuggingFace datasets streaming.
'''))

    # Cell 6: Data Preparation Code
    cells.append(nbf.v4.new_code_cell('''# Data configuration - no pre-download needed, data streams during training
# The vec2vec repo handles data loading via HuggingFace datasets

# Verify we can import the data utilities
import sys
sys.path.insert(0, '/content/vec2vec')

from utils.streaming_utils import load_streaming_embeddings
from utils.model_utils import load_encoder, get_sentence_embedding_dimension

# Test data loading
print("Testing data loading...")
try:
    dset = load_streaming_embeddings("nq")
    print(f"✓ NQ dataset loaded successfully")
    print(f"  Dataset features: {list(dset.features.keys())[:5]}...")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    print("  Will attempt during training...")

# Check embedding model dimensions
print(f"\\nEmbedding dimensions:")
print(f"  stella: {get_sentence_embedding_dimension('stella')}")
print(f"  gte: {get_sentence_embedding_dimension('gte')}")
'''))

    # Cell 7: VSP Implementation Overview
    cells.append(nbf.v4.new_markdown_cell('''## VSP Loss Implementation Overview

### Original VSP (from paper)
Located in `utils/train_utils.py`, the original VSP preserves pairwise similarities:

```python
# Normalize embeddings
B = target / target.norm(dim=1, keepdim=True)  # Target space
A = translated / translated.norm(dim=1, keepdim=True)  # Translated

# Compute similarity matrices
S_target = B @ B.T
S_translated = A @ A.T
S_cross = A @ B.T

# VSP loss = MAE between similarity matrices
vsp_loss = |S_target - S_translated|.mean() + |S_target - S_cross|.mean()
```

### Our Variants

1. **Conformal VSP**: Preserves angles (cosine similarities) more explicitly
   - Normalizes similarity matrices before comparison
   - Focuses on angular relationships

2. **Topological VSP (kNN)**: Preserves neighborhood structure
   - For each point, computes k-nearest neighbors in both spaces
   - Penalizes disagreement in neighbor sets (Jaccard similarity)

3. **Topological VSP (Soft)**: Soft neighborhood preservation
   - Converts distances to probability distributions
   - Minimizes KL divergence between neighbor distributions
'''))

    # Cell 8: VSP Variants Implementation
    vsp_code = '''# Create extended VSP loss functions with multiple variants

vsp_variants_code = """
import torch
import torch.nn.functional as F

def vsp_loss_original(ins, translations, logger=None) -> torch.Tensor:
    # Original VSP loss from paper - preserves pairwise similarities.
    loss = None
    EPS = 1e-10
    count = 0

    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)
            in_sims = B @ B.T

            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)
            out_sims = A @ A.T
            out_sims_reflected = A @ B.T

            vsp_loss = (in_sims - out_sims).abs().mean()
            vsp_loss_reflected = (in_sims - out_sims_reflected).abs().mean()

            if loss is None:
                loss = vsp_loss + vsp_loss_reflected
            else:
                loss += vsp_loss + vsp_loss_reflected
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0)


def vsp_loss_conformal(ins, translations, logger=None) -> torch.Tensor:
    # Conformal VSP - preserves angles (cosine similarities) explicitly.
    loss = None
    EPS = 1e-10
    count = 0

    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)

            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)

            cos_sim_target = B @ B.T
            cos_sim_trans = A @ A.T

            cos_sim_target_norm = cos_sim_target / (cos_sim_target.abs().max() + EPS)
            cos_sim_trans_norm = cos_sim_trans / (cos_sim_trans.abs().max() + EPS)

            conformal_loss = F.mse_loss(cos_sim_trans_norm, cos_sim_target_norm)

            cos_sim_cross = A @ B.T
            cos_sim_cross_norm = cos_sim_cross / (cos_sim_cross.abs().max() + EPS)
            cross_loss = F.mse_loss(cos_sim_cross_norm, cos_sim_target_norm)

            if loss is None:
                loss = conformal_loss + cross_loss
            else:
                loss += conformal_loss + cross_loss
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0)


def vsp_loss_topo_knn(ins, translations, logger=None, k=5) -> torch.Tensor:
    # Topological VSP (kNN) - preserves k-nearest neighbor structure.
    loss = None
    EPS = 1e-10
    count = 0

    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)

            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)

            batch_size = B.shape[0]
            k_actual = min(k, batch_size - 1)

            sim_target = B @ B.T
            sim_trans = A @ A.T

            mask = torch.eye(batch_size, device=B.device).bool()
            sim_target = sim_target.masked_fill(mask, -float('inf'))
            sim_trans = sim_trans.masked_fill(mask, -float('inf'))

            _, knn_target = sim_target.topk(k_actual, dim=1)
            _, knn_trans = sim_trans.topk(k_actual, dim=1)

            jaccard_sum = 0.0
            for i in range(batch_size):
                set_target = set(knn_target[i].tolist())
                set_trans = set(knn_trans[i].tolist())
                intersection = len(set_target & set_trans)
                union = len(set_target | set_trans)
                jaccard = intersection / union if union > 0 else 1.0
                jaccard_sum += jaccard

            topo_loss = 1.0 - (jaccard_sum / batch_size)

            if loss is None:
                loss = torch.tensor(topo_loss, device=A.device)
            else:
                loss = loss + topo_loss
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=next(iter(ins.values())).device)


def vsp_loss_topo_soft(ins, translations, logger=None, temperature=0.1) -> torch.Tensor:
    # Soft topological VSP - preserves neighborhood distributions via KL divergence.
    loss = None
    EPS = 1e-10
    count = 0

    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)

            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)

            batch_size = B.shape[0]

            sim_target = B @ B.T
            sim_trans = A @ A.T

            mask = torch.eye(batch_size, device=B.device).bool()
            sim_target = sim_target.masked_fill(mask, -float('inf'))
            sim_trans = sim_trans.masked_fill(mask, -float('inf'))

            prob_target = F.softmax(sim_target / temperature, dim=1)
            prob_trans = F.softmax(sim_trans / temperature, dim=1)

            kl_loss = F.kl_div(prob_trans.log(), prob_target, reduction='batchmean')

            if loss is None:
                loss = kl_loss
            else:
                loss += kl_loss
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0)


VSP_LOSS_REGISTRY = {
    'original': vsp_loss_original,
    'conformal': vsp_loss_conformal,
    'topo_knn': vsp_loss_topo_knn,
    'topo_soft': vsp_loss_topo_soft,
}

def get_vsp_loss_fn(vsp_type='original'):
    # Get VSP loss function by type.
    if vsp_type not in VSP_LOSS_REGISTRY:
        raise ValueError(f"Unknown VSP type: {vsp_type}. Available: {list(VSP_LOSS_REGISTRY.keys())}")
    return VSP_LOSS_REGISTRY[vsp_type]
"""

# Save the VSP variants to a file
with open('/content/vec2vec/utils/vsp_variants.py', 'w') as f:
    f.write(vsp_variants_code)

print("Created utils/vsp_variants.py with VSP loss variants:")
print("  - original: Standard dot-product preservation")
print("  - conformal: Angle/cosine similarity preservation")
print("  - topo_knn: k-nearest neighbor preservation")
print("  - topo_soft: Soft neighborhood distribution preservation")
'''
    cells.append(nbf.v4.new_code_cell(vsp_code))

    # Cell 9: Modified Training Script
    training_script_cell = '''# Create a modified training script that supports VSP variants

training_script = """
#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
import toml
from tqdm import tqdm
import json

sys.path.insert(0, '/content/vec2vec')

from utils.utils import load_n_translator
from utils.model_utils import load_encoder, get_sentence_embedding_dimension
from utils.streaming_utils import load_streaming_embeddings, MultiencoderTokenizedDataset, process_batch
from utils.collate import TokenizedCollator
from utils.train_utils import rec_loss_fn
from utils.vsp_variants import get_vsp_loss_fn
from utils.gan import LeastSquaresGAN
from translators.Discriminator import Discriminator


def create_config(vsp_type='original', num_points=50000, batch_size=64, epochs=3, lr=2e-5):
    return {
        'seed': 42,
        'dataset': 'nq',
        'unsup_emb': 'stella',
        'sup_emb': 'gte',
        'num_points': num_points,
        'val_size': 4096,
        'normalize_embeddings': True,
        'mixed_precision': 'fp16',
        'style': 'res_mlp',
        'depth': 2,
        'transform_depth': 3,
        'd_adapter': 512,
        'd_hidden': 512,
        'norm_style': 'batch',
        'gan_style': 'least_squares',
        'disc_depth': 3,
        'disc_dim': 256,
        'bs': batch_size,
        'lr': lr,
        'disc_lr': lr,
        'epochs': epochs,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'loss_coefficient_rec': 1.0,
        'loss_coefficient_vsp': 1.0,
        'loss_coefficient_gen': 0.5,
        'vsp_type': vsp_type,
    }


def train_vec2vec(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
    )

    device = accelerator.device
    print(f"Training on device: {device}")
    print(f"VSP type: {config['vsp_type']}")

    print("\\\\nLoading embedding models...")
    encoders = {}
    encoders[config['unsup_emb']] = load_encoder(config['unsup_emb'])
    encoders[config['sup_emb']] = load_encoder(config['sup_emb'])

    for name, encoder in encoders.items():
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    unsup_dim = get_sentence_embedding_dimension(config['unsup_emb'])
    sup_dim = get_sentence_embedding_dimension(config['sup_emb'])

    print("\\\\nCreating translator...")
    from translators.TransformTranslator import TransformTranslator

    translator = TransformTranslator(
        dims={config['unsup_emb']: unsup_dim, config['sup_emb']: sup_dim},
        adapter_depth=config['depth'],
        d_adapter=config['d_adapter'],
        d_hidden=config['d_hidden'],
        n_style=config['style'],
        transform_depth=config['transform_depth'],
        norm_style=config['norm_style'],
    )
    translator = translator.to(device)

    discriminator = Discriminator(
        dims={config['unsup_emb']: unsup_dim, config['sup_emb']: sup_dim},
        depth=config['disc_depth'],
        hidden_dim=config['disc_dim'],
    )
    discriminator = discriminator.to(device)

    gan = LeastSquaresGAN()

    optimizer_g = torch.optim.AdamW(translator.parameters(), lr=config['lr'])
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=config['disc_lr'])

    vsp_loss_fn = get_vsp_loss_fn(config['vsp_type'])

    print("\\\\nLoading data...")
    dset = load_streaming_embeddings(config['dataset'])
    dset_dict = dset.train_test_split(test_size=config['val_size'], seed=42)
    train_dset = dset_dict["train"]

    train_dset = train_dset.select(range(config['num_points']))

    train_tokenized = MultiencoderTokenizedDataset(
        train_dset, encoders,
        n_embs_per_batch=1,
        batch_size=config['bs'],
        seed=config['seed']
    )

    train_loader = DataLoader(
        train_tokenized,
        batch_size=1,
        num_workers=0,
        collate_fn=TokenizedCollator(),
    )

    translator, discriminator, optimizer_g, optimizer_d, train_loader = accelerator.prepare(
        translator, discriminator, optimizer_g, optimizer_d, train_loader
    )

    print(f"\\\\nStarting training for {config['epochs']} epochs...")
    print(f"Training samples: {config['num_points']}")
    print(f"Batch size: {config['bs']}")

    history = {'loss': [], 'rec_loss': [], 'vsp_loss': [], 'gen_loss': [], 'disc_loss': []}

    global_step = 0
    for epoch in range(config['epochs']):
        translator.train()
        discriminator.train()

        epoch_losses = {'loss': 0, 'rec_loss': 0, 'vsp_loss': 0, 'gen_loss': 0, 'disc_loss': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            ins = process_batch(batch, encoders, config['normalize_embeddings'], device)
            recs, translations = translator(ins)

            optimizer_d.zero_grad()
            disc_real = discriminator(ins)
            disc_fake = discriminator({k: v[config['unsup_emb']] for k, v in translations.items()})
            disc_loss = gan.discriminator_loss(disc_real, disc_fake)
            accelerator.backward(disc_loss)
            optimizer_d.step()

            optimizer_g.zero_grad()
            rec_loss = rec_loss_fn(ins, recs, None)
            vsp_loss = vsp_loss_fn(ins, translations, None)
            disc_fake_for_gen = discriminator({k: v[config['unsup_emb']] for k, v in translations.items()})
            gen_loss = gan.generator_loss(disc_fake_for_gen)

            total_loss = (
                config['loss_coefficient_rec'] * rec_loss +
                config['loss_coefficient_vsp'] * vsp_loss +
                config['loss_coefficient_gen'] * gen_loss
            )

            accelerator.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(translator.parameters(), config['max_grad_norm'])
            optimizer_g.step()

            epoch_losses['loss'] += total_loss.item()
            epoch_losses['rec_loss'] += rec_loss.item()
            epoch_losses['vsp_loss'] += vsp_loss.item() if isinstance(vsp_loss, torch.Tensor) else vsp_loss
            epoch_losses['gen_loss'] += gen_loss.item()
            epoch_losses['disc_loss'] += disc_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'rec': f"{rec_loss.item():.4f}",
                'vsp': f"{vsp_loss.item() if isinstance(vsp_loss, torch.Tensor) else vsp_loss:.4f}",
            })

            global_step += 1

        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            history[key].append(epoch_losses[key])

        print(f"Epoch {epoch+1} - Loss: {epoch_losses['loss']:.4f}, "
              f"Rec: {epoch_losses['rec_loss']:.4f}, "
              f"VSP: {epoch_losses['vsp_loss']:.4f}")

    print(f"\\\\nSaving model to {output_dir}")
    torch.save({
        'translator_state_dict': accelerator.unwrap_model(translator).state_dict(),
        'discriminator_state_dict': accelerator.unwrap_model(discriminator).state_dict(),
        'config': config,
        'history': history,
    }, os.path.join(output_dir, 'checkpoint.pt'))

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)

    print("Training complete!")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vsp_type', type=str, default='original',
                        choices=['original', 'conformal', 'topo_knn', 'topo_soft'])
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    config = create_config(
        vsp_type=args.vsp_type,
        num_points=args.num_points,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    train_vec2vec(config, args.output_dir)
"""

with open('/content/vec2vec/train_vsp_variants.py', 'w') as f:
    f.write(training_script)

print("✓ Created train_vsp_variants.py - modified training script with VSP variant support")
'''
    cells.append(nbf.v4.new_code_cell(training_script_cell))

    # Cell 10: Training Configuration Explanation
    cells.append(nbf.v4.new_markdown_cell('''## Training Configuration

### Hyperparameters for Colab (reduced for single GPU)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Training samples | 50,000 | Subset of full dataset |
| Validation samples | 4,096 | For evaluation |
| Batch size | 64 | Fits in T4/L4 GPU memory |
| Epochs | 3 | Quick iteration; increase for better results |
| Learning rate | 2e-5 | Default from paper |
| Model pair | stella → gte | 768-dim to 768-dim |
| Architecture | res_mlp | Residual MLP transform |
| Mixed precision | fp16 | Memory efficiency |

### Training Runs

We'll train three models:
1. **Original VSP** → `outputs/vec2vec_original/`
2. **Conformal VSP** → `outputs/vec2vec_conformal/`
3. **Topological kNN VSP** → `outputs/vec2vec_topo_knn/`

Each run takes approximately 20-40 minutes on a T4 GPU.
'''))

    # Cell 11: Show/Define Configs
    cells.append(nbf.v4.new_code_cell('''# Training configuration
import json

# Common parameters
COMMON_CONFIG = {
    'num_points': 50000,   # Training samples (increase to 100k+ for better results)
    'batch_size': 64,      # Batch size (reduce if OOM)
    'epochs': 3,           # Training epochs (increase to 10+ for better results)
    'lr': 2e-5,            # Learning rate
}

# VSP variants to train
VSP_VARIANTS = ['original', 'conformal', 'topo_knn']

# Display configurations
print("Training Configuration")
print("=" * 50)
print(f"Common parameters: {json.dumps(COMMON_CONFIG, indent=2)}")
print(f"\\nVSP variants to train: {VSP_VARIANTS}")
print("\\nOutput directories:")
for vsp_type in VSP_VARIANTS:
    print(f"  - {vsp_type}: outputs/vec2vec_{vsp_type}/")

print("\\n" + "=" * 50)
print("To adjust training:")
print("  - Increase num_points for more data")
print("  - Increase epochs for better convergence")
print("  - Reduce batch_size if out of memory")
print("  - Comment out VSP variants in VSP_VARIANTS to skip")
'''))

    # Cell 12: Launch Training
    cells.append(nbf.v4.new_code_cell('''# Train all VSP variants
import os
import subprocess
import time

# Training parameters (can modify above)
num_points = COMMON_CONFIG['num_points']
batch_size = COMMON_CONFIG['batch_size']
epochs = COMMON_CONFIG['epochs']
lr = COMMON_CONFIG['lr']

# Train each variant
results = {}

for vsp_type in VSP_VARIANTS:
    print("\\n" + "=" * 60)
    print(f"Training vec2vec with VSP type: {vsp_type}")
    print("=" * 60)

    output_dir = f"/content/vec2vec/outputs/vec2vec_{vsp_type}"

    start_time = time.time()

    # Run training
    cmd = [
        "python", "/content/vec2vec/train_vsp_variants.py",
        "--vsp_type", vsp_type,
        "--num_points", str(num_points),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--output_dir", output_dir,
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/content/vec2vec")

    elapsed = time.time() - start_time
    results[vsp_type] = {
        'output_dir': output_dir,
        'elapsed_time': elapsed,
        'success': result.returncode == 0
    }

    print(f"\\nCompleted {vsp_type} in {elapsed/60:.1f} minutes")
    print(f"Output saved to: {output_dir}")

print("\\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
for vsp_type, info in results.items():
    status = "✓" if info['success'] else "✗"
    print(f"{status} {vsp_type}: {info['elapsed_time']/60:.1f} min - {info['output_dir']}")
'''))

    # Cell 13: Evaluation Explanation
    cells.append(nbf.v4.new_markdown_cell('''## Evaluation

We evaluate each trained model on a held-out test set using:

### Metrics

1. **Cosine Similarity**
   - Average cosine similarity between translated embeddings and ground-truth target embeddings
   - Range: [-1, 1], higher is better
   - Measures direct alignment quality

2. **Top-1 Accuracy**
   - For each translated embedding, find the nearest neighbor in the target embedding space
   - Check if the nearest neighbor is the correct corresponding embedding
   - Range: [0, 1], higher is better
   - Measures retrieval accuracy

3. **Mean Rank**
   - For each translated embedding, rank all target embeddings by similarity
   - Report the average rank of the correct target
   - Lower is better (1 = perfect)
   - Measures how well the translation preserves relative positions
'''))

    # Cell 14: Evaluation Code
    eval_code = '''# Evaluation script
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

import sys
sys.path.insert(0, '/content/vec2vec')

from utils.model_utils import load_encoder, get_sentence_embedding_dimension
from utils.streaming_utils import load_streaming_embeddings, MultiencoderTokenizedDataset, process_batch
from utils.collate import TokenizedCollator
from torch.utils.data import DataLoader
from translators.TransformTranslator import TransformTranslator


def evaluate_model(checkpoint_path, test_size=8192, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    print(f"\\nEvaluating model: {checkpoint_path}")
    print(f"VSP type: {config['vsp_type']}")

    encoders = {}
    encoders[config['unsup_emb']] = load_encoder(config['unsup_emb'])
    encoders[config['sup_emb']] = load_encoder(config['sup_emb'])

    for encoder in encoders.values():
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    unsup_dim = get_sentence_embedding_dimension(config['unsup_emb'])
    sup_dim = get_sentence_embedding_dimension(config['sup_emb'])

    translator = TransformTranslator(
        dims={config['unsup_emb']: unsup_dim, config['sup_emb']: sup_dim},
        adapter_depth=config['depth'],
        d_adapter=config['d_adapter'],
        d_hidden=config['d_hidden'],
        n_style=config['style'],
        transform_depth=config['transform_depth'],
        norm_style=config['norm_style'],
    )
    translator.load_state_dict(checkpoint['translator_state_dict'])
    translator = translator.to(device)
    translator.eval()

    dset = load_streaming_embeddings(config['dataset'])
    dset_dict = dset.train_test_split(test_size=test_size + config['val_size'], seed=42)
    test_dset = dset_dict["test"].select(range(test_size))

    test_tokenized = MultiencoderTokenizedDataset(
        test_dset, encoders,
        n_embs_per_batch=1,
        batch_size=batch_size,
        seed=42
    )

    test_loader = DataLoader(
        test_tokenized,
        batch_size=1,
        num_workers=0,
        collate_fn=TokenizedCollator(),
    )

    all_source = []
    all_target = []
    all_translated = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test data"):
            ins = process_batch(batch, encoders, config['normalize_embeddings'], device)

            source_emb = ins[config['unsup_emb']]
            target_emb = ins[config['sup_emb']]

            _, translations = translator(ins)
            translated_emb = translations[config['sup_emb']][config['unsup_emb']]

            all_source.append(source_emb.cpu())
            all_target.append(target_emb.cpu())
            all_translated.append(translated_emb.cpu())

    all_source = torch.cat(all_source, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_translated = torch.cat(all_translated, dim=0)

    print(f"Test samples: {all_source.shape[0]}")

    # 1. Cosine similarity
    cosine_sims = F.cosine_similarity(all_translated, all_target, dim=1)
    mean_cosine = cosine_sims.mean().item()

    # 2. Top-1 accuracy and rank
    translated_norm = F.normalize(all_translated, dim=1)
    target_norm = F.normalize(all_target, dim=1)

    similarity_matrix = translated_norm @ target_norm.T

    top1_predictions = similarity_matrix.argmax(dim=1)
    correct_indices = torch.arange(all_translated.shape[0])
    top1_accuracy = (top1_predictions == correct_indices).float().mean().item()

    ranks = []
    for i in range(similarity_matrix.shape[0]):
        sims = similarity_matrix[i]
        sorted_indices = sims.argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    mean_rank = np.mean(ranks)

    results = {
        'vsp_type': config['vsp_type'],
        'train_size': config['num_points'],
        'test_size': all_source.shape[0],
        'cosine_similarity': mean_cosine,
        'top1_accuracy': top1_accuracy,
        'mean_rank': mean_rank,
    }

    print(f"Results for {config['vsp_type']}:")
    print(f"  Cosine Similarity: {mean_cosine:.4f}")
    print(f"  Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"  Mean Rank: {mean_rank:.2f}")

    return results


# Evaluate all trained models
all_results = []

for vsp_type in VSP_VARIANTS:
    checkpoint_path = f"/content/vec2vec/outputs/vec2vec_{vsp_type}/checkpoint.pt"

    if os.path.exists(checkpoint_path):
        try:
            result = evaluate_model(checkpoint_path, test_size=8192, batch_size=128)
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {vsp_type}: {e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

# Create results table
if all_results:
    df = pd.DataFrame(all_results)
    df = df[['vsp_type', 'train_size', 'test_size', 'cosine_similarity', 'top1_accuracy', 'mean_rank']]
    df.columns = ['VSP Type', 'Train Size', 'Test Size', 'Cosine Sim', 'Top-1 Acc', 'Mean Rank']

    print("\\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    df.to_csv('/content/vec2vec/outputs/vsp_comparison_results.csv', index=False)
    print("\\nResults saved to outputs/vsp_comparison_results.csv")
else:
    print("No models were successfully evaluated.")
'''
    cells.append(nbf.v4.new_code_cell(eval_code))

    # Cell 15: Visualization
    viz_code = '''# Visualization of results
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

if all_results:
    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = sns.color_palette("husl", len(df))

    # 1. Cosine Similarity
    ax = axes[0]
    bars = ax.bar(df['vsp_type'], df['cosine_similarity'], color=colors, edgecolor='black')
    ax.set_xlabel('VSP Type', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity (Higher is Better)', fontsize=14)
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, df['cosine_similarity']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. Top-1 Accuracy
    ax = axes[1]
    bars = ax.bar(df['vsp_type'], df['top1_accuracy'], color=colors, edgecolor='black')
    ax.set_xlabel('VSP Type', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy', fontsize=12)
    ax.set_title('Top-1 Accuracy (Higher is Better)', fontsize=14)
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, df['top1_accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # 3. Mean Rank
    ax = axes[2]
    bars = ax.bar(df['vsp_type'], df['mean_rank'], color=colors, edgecolor='black')
    ax.set_xlabel('VSP Type', fontsize=12)
    ax.set_ylabel('Mean Rank', fontsize=12)
    ax.set_title('Mean Rank (Lower is Better)', fontsize=14)
    for bar, val in zip(bars, df['mean_rank']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('/content/vec2vec/outputs/vsp_comparison_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\\nPlot saved to outputs/vsp_comparison_plot.png")

    # Plot training curves if available
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    for vsp_type in VSP_VARIANTS:
        history_path = f"/content/vec2vec/outputs/vec2vec_{vsp_type}/history.json"
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

            epochs = range(1, len(history['loss']) + 1)

            axes2[0].plot(epochs, history['loss'], marker='o', label=vsp_type)
            axes2[1].plot(epochs, history['vsp_loss'], marker='o', label=vsp_type)

    axes2[0].set_xlabel('Epoch')
    axes2[0].set_ylabel('Total Loss')
    axes2[0].set_title('Training Loss')
    axes2[0].legend()

    axes2[1].set_xlabel('Epoch')
    axes2[1].set_ylabel('VSP Loss')
    axes2[1].set_title('VSP Loss')
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig('/content/vec2vec/outputs/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training curves saved to outputs/training_curves.png")
else:
    print("No results to visualize.")
'''
    cells.append(nbf.v4.new_code_cell(viz_code))

    # Cell 16: Summary and Discussion
    cells.append(nbf.v4.new_markdown_cell('''## Summary & Discussion

### What This Notebook Reproduces

This is a **scaled-down reproduction** of vec2vec with VSP variants:
- **Model pair**: stella → gte (768-dim text embeddings)
- **Dataset**: Natural Questions (NQ)
- **Training size**: ~50,000 examples (full paper uses 100k+)
- **Epochs**: 3 (increase for better results)

### Interpreting Results

Compare the three VSP variants:

1. **Original VSP** (baseline)
   - Preserves raw pairwise similarities
   - What the paper uses

2. **Conformal VSP**
   - Preserves angles/cosine similarities more explicitly
   - May help if the embedding spaces have different scales

3. **Topological kNN VSP**
   - Preserves neighborhood structure
   - More flexible - does not require exact similarity preservation
   - May help with noisy or sparse embeddings

### Key Questions to Answer

- **Does conformal VSP improve upon original?**
  - If yes: angle preservation matters more than raw dot products

- **Does topological VSP improve upon original?**
  - If yes: neighborhood structure is more important than exact similarities

- **Trade-offs between metrics?**
  - A method might improve Top-1 but hurt Mean Rank, or vice versa

### How to Improve Results

1. **Increase training data**: `num_points = 100000` or more
2. **More epochs**: `epochs = 10` or more
3. **Tune hyperparameters**: Learning rate, batch size, model capacity
4. **Try other model pairs**: gte→gtr, e5→gtr, etc.
5. **Adjust kNN k**: Try k=3, k=10 for topological VSP
6. **Adjust temperature**: For topo_soft variant

### Code Locations

| Component | Location |
|-----------|----------|
| VSP variants | `utils/vsp_variants.py` |
| Training script | `train_vsp_variants.py` |
| Results | `outputs/vsp_comparison_results.csv` |
| Plots | `outputs/vsp_comparison_plot.png` |

### Next Steps

1. Run with more data and epochs
2. Try additional VSP variants (topo_soft)
3. Experiment with different model pairs
4. Compare with optimal transport baseline
5. Analyze per-sample results for failure cases
'''))

    # Cell 17: Final Summary
    cells.append(nbf.v4.new_code_cell('''# Final summary
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print("\\nFiles created:")
print("  - utils/vsp_variants.py (VSP loss implementations)")
print("  - train_vsp_variants.py (training script)")
print("  - outputs/vec2vec_*/checkpoint.pt (trained models)")
print("  - outputs/vsp_comparison_results.csv (evaluation results)")
print("  - outputs/*.png (visualization plots)")

print("\\nTo re-run with different settings:")
print("  1. Modify COMMON_CONFIG in cell 11")
print("  2. Modify VSP_VARIANTS to add/remove variants")
print("  3. Re-run cells 12-15")

print("\\nTo use a trained model:")
print("  checkpoint = torch.load('outputs/vec2vec_original/checkpoint.pt')")
print("  translator.load_state_dict(checkpoint['translator_state_dict'])")
'''))

    # Add cells to notebook
    nb['cells'] = cells

    # Set notebook metadata
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        },
        'accelerator': 'GPU',
        'colab': {
            'provenance': [],
            'gpuType': 'T4'
        }
    }

    return nb


if __name__ == '__main__':
    notebook = create_notebook()

    output_path = '/home/user/vec2vec/vec2vec_colab_50k_vsp_variants.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)

    print(f"Notebook created: {output_path}")
    print(f"Cells: {len(notebook['cells'])}")
