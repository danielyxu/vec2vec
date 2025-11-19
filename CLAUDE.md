# CLAUDE.md - AI Assistant Guide for Vec2Vec

## Project Overview

Vec2Vec is a research framework for training Generative Adversarial Networks (GANs) to convert between different embedding models while preserving semantic relationships. The framework enables transformation of embeddings from one latent space to another (e.g., from unsupervised models like GTE to supervised models like GTR).

**Paper:** "Harnessing the Universal Geometry of Embeddings" (ArXiv: 2505.12540)
**Website:** https://vec2vec.github.io/

## Quick Reference

### Common Commands

```bash
# Training with config
python train.py configs/unsupervised --epochs 10

# Training with overrides
python train.py configs/unsupervised --num_points 100000 --batch_size 32

# Evaluation
python eval.py [model_path]

# Optimal transport baseline
python ot_baseline.py configs/ot
```

### Key Files to Understand First

1. `train.py` - Main training loop, start here to understand workflow
2. `configs/unsupervised.toml` - Example config showing all parameters
3. `translators/TransformTranslator.py` - Primary translator architecture
4. `utils/gan.py` - GAN implementations (Vanilla, LeastSquares, Relativistic)
5. `utils/train_utils.py` - Loss functions

## Directory Structure

```
vec2vec/
├── configs/                    # TOML experiment configurations
│   ├── unsupervised.toml      # Main training config
│   └── ot.toml                # Optimal transport baseline
├── translators/               # Neural network architectures
│   ├── AbsNTranslator.py      # Abstract base class
│   ├── TransformTranslator.py # Main translator (adapters + transform)
│   ├── LinearTranslator.py    # Simple linear mapping
│   ├── MLPWithResidual.py     # MLP with residual connections
│   ├── Discriminator.py       # GAN discriminator
│   ├── IdentityBaseline.py    # Identity baseline
│   └── transforms/            # Transform modules
│       ├── AbsTransform.py    # Abstract transform base
│       ├── UNetTransform.py   # 2D U-Net
│       └── UNet1dTransform.py # 1D U-Net
├── utils/                     # Utilities and helpers
│   ├── gan.py                 # GAN implementations
│   ├── train_utils.py         # Loss functions
│   ├── eval_utils.py          # Evaluation metrics
│   ├── model_utils.py         # Model loading utilities
│   ├── utils.py               # Translator loading, HF integration
│   ├── collate.py             # Data batching
│   ├── streaming_utils.py     # Streaming data loading
│   ├── wandb_logger.py        # W&B integration
│   └── dist.py                # Distributed training
├── labels/                    # Label data for experiments
├── train.py                   # Main training script
├── eval.py                    # Evaluation script
├── universal.py               # HF model loading utilities
├── ot_baseline.py             # Optimal transport baseline
└── environment.yml            # Conda environment
```

## Architecture Patterns

### Translator Hierarchy

All translators inherit from `AbsNTranslator`:

- **TransformTranslator** (Primary): Input adapters -> Transform -> Output adapters
- **LinearTranslator**: Direct linear mapping
- **MLPWithResidual**: MLP with skip connections
- **IdentityBaseline**: Returns input unchanged

### Transform Styles

Load via `utils.py::load_n_translator()`:

- `'linear'` - Single linear layer
- `'n_simple'` - Linear transform
- `'n_double'` - SiLU + Linear
- `'res_mlp'` - Residual MLP
- `'n_ae'` - Autoencoder bottleneck
- `'unet'` / `'unet1d'` - U-Net transforms

### GAN Variants

Three implementations in `utils/gan.py`:

1. **VanillaGAN** - BCE with label smoothing
2. **LeastSquaresGAN** - MSE-based, more stable
3. **RelativisticGAN** - Comparative discriminator

## Configuration System

### TOML Structure

```toml
[general]
seed = 42
unsup_emb = "gte"          # Source embedding model
sup_emb = "gtr"            # Target embedding model
num_points = 100000        # Training samples

[translator]
style = "res_mlp"          # Architecture type
depth = 2                  # Adapter layer count
d_adapter = 512            # Adapter dimensions
d_hidden = 1024            # Hidden dimensions
norm_style = "layer"       # Normalization type

[discriminator]
gan_style = "least_squares"
disc_depth = 4
disc_dim = 512

[train]
batch_size = 512
lr = 1e-4
epochs = 100
rec_coeff = 1.0            # Reconstruction loss weight
trans_coeff = 1.0          # Translation loss weight
vsp_coeff = 0.1            # Vector set preservation weight

[gan]
disc_lr = 1e-4
smooth = 0.9               # Label smoothing

[eval]
val_size = 10000
eval_batch_size = 256

[logging]
use_wandb = true
save_dir = "results"
```

### Command-Line Overrides

Any config value can be overridden:

```bash
python train.py configs/unsupervised \
    --num_points 50000 \
    --batch_size 256 \
    --lr 5e-5 \
    --gan_style vanilla
```

## Key Loss Functions

Located in `utils/train_utils.py`:

- **`rec_loss_fn`**: Reconstruction loss (1 - cosine_similarity)
- **`trans_loss_fn`**: Translation between embedding spaces
- **`vsp_loss_fn`**: Vector Set Preservation - maintains similarity structure
- **`contrastive_loss_fn`**: Contrastive learning objective

## Development Conventions

### Code Style

- Python 3.10+ with type hints where practical
- PyTorch for all neural network code
- Use `accelerate` for distributed training
- TOML for configuration files
- SafeTensors for model serialization

### Naming Conventions

- Translators: PascalCase with `Translator` suffix
- Utils: snake_case functions
- Config keys: snake_case
- Embedding models: lowercase abbreviations (gte, gtr, stella)

### Model I/O

```python
# Saving models
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")

# Loading from HuggingFace
from utils.utils import load_translator_from_hf
translator = load_translator_from_hf("username/model-name")
```

### Data Handling

- Use streaming for large embedding datasets
- Batch loading via `MultiencoderTokenizedDataset`
- Custom collation with `TokenizedCollator`

## Important Considerations

### When Modifying Translators

1. Inherit from `AbsNTranslator`
2. Implement `forward()` method
3. Handle both single and batch inputs
4. Consider adding to `load_n_translator()` switch statement

### When Adding Loss Functions

1. Add to `utils/train_utils.py`
2. Follow signature: `fn(pred, target, **kwargs) -> Tensor`
3. Return scalar tensor for backward compatibility
4. Add corresponding coefficient in config

### When Modifying Training Loop

1. Main loop is in `train.py`
2. Uses HuggingFace `accelerate` for distribution
3. Gradient accumulation is built-in
4. Early stopping via patience parameter

### Mixed Precision

- FP16/BF16 support via accelerate
- Automatic gradient scaling
- Set via config: `precision = "fp16"` or `"bf16"`

## Testing & Validation

This is a research codebase without formal tests. Validation happens through:

- Early stopping with patience
- Validation set evaluation during training
- k-NN retrieval metrics
- Similarity preservation checks

Run evaluation after training:

```bash
python eval.py path/to/saved/model
```

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate vec2vec

# Or install key dependencies manually
pip install torch transformers datasets accelerate wandb safetensors toml
```

## Common Tasks

### Adding a New Embedding Model

1. Add model name mapping in `utils/model_utils.py`
2. Update `load_encoder()` function
3. Add dimension info to `get_sentence_embedding_dimension()`

### Creating a New Experiment Config

1. Copy `configs/unsupervised.toml`
2. Modify parameters as needed
3. Run with: `python train.py configs/your_config`

### Debugging Training Issues

1. Check W&B logs if enabled
2. Monitor gradient norms via `get_grad_norm()`
3. Verify loss coefficients are balanced
4. Check discriminator vs generator loss ratio

### Resuming Training

Load checkpoint and continue:

```python
# In train.py, checkpoints are saved to save_dir
# Resume by loading state dict before training loop
```

## File Locations for Common Edits

| Task | Primary File | Secondary Files |
|------|-------------|-----------------|
| Add translator | `translators/` | `utils/utils.py` |
| Add loss function | `utils/train_utils.py` | `train.py` |
| Modify GAN | `utils/gan.py` | `train.py` |
| Add metric | `utils/eval_utils.py` | `eval.py` |
| Change data loading | `utils/collate.py` | `utils/streaming_utils.py` |
| Modify config schema | `configs/*.toml` | `train.py` (argument parsing) |

## Dependencies

Core:
- PyTorch 2.0+
- Transformers 4.29+
- Datasets 2.12+
- Accelerate (distributed training)
- WandB (experiment tracking)

See `environment.yml` for complete environment specification.

## Git Workflow

- Main branch: `main`
- Development on feature branches
- Commit messages: descriptive, present tense
- Keep configs and results separate from code

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Enable gradient accumulation
- Use mixed precision (`precision = "fp16"`)

### Training Instability

- Try `LeastSquaresGAN` instead of `VanillaGAN`
- Reduce learning rates
- Increase `disc_lr` relative to generator lr
- Enable gradient clipping

### Poor Convergence

- Increase `vsp_coeff` for better similarity preservation
- Adjust loss coefficient balance
- Try different translator `style`
- Increase model capacity (`d_hidden`, `depth`)
