import multiprocessing
import os, json
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub.file_download import hf_hub_download
from safetensors.torch import load_file

from translators.IdentityBaseline import IdentityBaseline
from translators.MLPWithResidual import MLPWithResidual
from translators.LinearTranslator import LinearTranslator
from translators.TransformTranslator import TransformTranslator
from translators.transforms.UNetTransform import UNetTransform
from translators.transforms.UNet1dTransform import UNet1dTransform

from vec2text.models import InversionModel


def load_n_translator(cfg, encoder_dims):
    if cfg.style == 'identity':
        return IdentityBaseline(encoder_dims)
    if cfg.style == 'linear':
        return LinearTranslator(
            encoder_dims,
            cfg.normalize_embeddings,
            cfg.src_emb if hasattr(cfg, 'src_emb') else None,
            cfg.tgt_emb if hasattr(cfg, 'tgt_emb') else None
        )

    if cfg.style == 'n_simple':
        transform = nn.Linear(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'n_double':
        transform = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
        )
    elif cfg.style == 'res_mlp':
        transform = MLPWithResidual(
            depth=cfg.transform_depth,
            in_dim=cfg.d_adapter, 
            hidden_dim=cfg.d_transform, 
            out_dim=cfg.d_adapter, 
            norm_style=cfg.norm_style,
            weight_init=cfg.weight_init,
        )
    elif cfg.style == 'n_ae':
        transform = nn.Sequential(
            nn.Linear(cfg.d_adapter, cfg.latent_dims),
            nn.ReLU(),
            nn.Linear(cfg.latent_dims, cfg.d_adapter)
        )
    elif cfg.style == 'unet':
        transform = UNetTransform(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'unet1d':
        transform = UNet1dTransform(cfg.d_adapter, cfg.d_adapter)
    else:
        raise ValueError(f"Unknown style: {cfg.style}")

    return TransformTranslator(
        encoder_dims=encoder_dims,
        d_adapter=cfg.d_adapter,
        d_hidden=cfg.d_hidden,
        transform=transform,
        weight_init=cfg.weight_init,
        depth=cfg.depth,
        use_small_output_adapters=cfg.use_small_output_adapters if hasattr(cfg, 'use_small_output_adapters') else False,
        norm_style=cfg.norm_style if hasattr(cfg, 'norm_style') else 'batch',
    )


def get_inverters(emb_flags, device='cpu'):
    assert isinstance(emb_flags, list)
    inverters = {}
    for emb_flag in emb_flags:
        assert emb_flag in ['gtr', 'gte']
        if emb_flag == "gtr":
            # inversion_model = InversionModel.from_pretrained("jxm/gtr-32-noise-0.001")
            # inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")
            inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")

        elif emb_flag == 'gte':
            inversion_model = InversionModel.from_pretrained("jxm/gte-32-noise-0.001")
        inversion_model.eval()
        inverters[emb_flag] = inversion_model.to(device)
    return inverters

def read_args(argv):
    cfg = {}
    # Handle unknown arguments
    for arg in argv:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            # Attempt to parse value as int, float, or leave as string
            try:
                value = int(argv[argv.index(arg) + 1])
            except ValueError:
                try:
                    value = float(argv[argv.index(arg) + 1])
                except ValueError:
                    value = argv[argv.index(arg) + 1]
            cfg[key] = value
    return cfg


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1

def get_num_proc() -> int:
    world_size: int = torch.cuda.device_count()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size
    

def load_translator_from_hf(model_id):
    if os.path.isdir(model_id):
        print("Loading weights from local directory")
        model_file = os.path.join(model_id, 'model.safetensors')
        config_file = os.path.join(model_id, 'config.json')
    else:
        model_file = hf_hub_download(
            repo_id=model_id,
            filename='model.safetensors',
        )
        config_file = hf_hub_download(
            repo_id=model_id,
            filename='config.json',
        )
    state_dict = load_file(model_file)
    with open(config_file) as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg)
    translator = load_n_translator(cfg, cfg.encoder_dims)
    translator.load_state_dict(state_dict, strict=False)
    return translator


def exit_on_nan(loss: torch.Tensor) -> None:
    if torch.isnan(loss).any():
        print("Loss is NaN! exiting")
        exit(1)


def save_everything(cfg, translator, opt, gans, save_dir):
    torch.save(translator.state_dict(), os.path.join(save_dir, 'model.pt'))
    torch.save(opt.state_dict(), os.path.join(save_dir, 'opt.pt'))
    for i, gan in enumerate(gans):
        torch.save(gan.discriminator.state_dict(), os.path.join(save_dir, f'gan_{i}.pt'))
        torch.save(gan.discriminator_opt.state_dict(), os.path.join(save_dir, f'gan_opt_{i}.pt'))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg.__dict__, f)


def save_checkpoint(
    save_dir,
    epoch,
    translator,
    opt,
    scheduler,
    gans,
    accelerator,
    early_stopper=None,
    best_score=None,
):
    """Save a complete checkpoint for resuming training.

    Args:
        save_dir: Directory to save checkpoint
        epoch: Current epoch number
        translator: The translator model
        opt: Optimizer for translator
        scheduler: Learning rate scheduler
        gans: List of GAN objects (each with discriminator, discriminator_opt, discriminator_scheduler)
        accelerator: Accelerator instance for unwrapping models
        early_stopper: Optional EarlyStopper instance
        best_score: Optional best validation score
    """
    checkpoint = {
        'epoch': epoch,
        'translator_state_dict': accelerator.unwrap_model(translator).state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
    }

    # Save early stopper state if present
    if early_stopper is not None:
        checkpoint['early_stopper'] = {
            'counter': early_stopper.counter,
            'opt_val': early_stopper.opt_val,
        }

    # Save discriminator states
    for i, gan in enumerate(gans):
        checkpoint[f'discriminator_{i}_state_dict'] = accelerator.unwrap_model(gan.discriminator).state_dict()
        checkpoint[f'discriminator_opt_{i}_state_dict'] = gan.discriminator_opt.state_dict()
        checkpoint[f'discriminator_scheduler_{i}_state_dict'] = gan.discriminator_scheduler.state_dict()

    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def load_checkpoint(
    save_dir,
    translator,
    opt,
    scheduler,
    gans,
    accelerator,
    early_stopper=None,
):
    """Load a checkpoint for resuming training.

    Args:
        save_dir: Directory containing checkpoint
        translator: The translator model
        opt: Optimizer for translator
        scheduler: Learning rate scheduler
        gans: List of GAN objects
        accelerator: Accelerator instance
        early_stopper: Optional EarlyStopper instance

    Returns:
        Tuple of (start_epoch, best_score) or (0, None) if no checkpoint found
    """
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, None

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load translator state
    accelerator.unwrap_model(translator).load_state_dict(checkpoint['translator_state_dict'])

    # Load optimizer and scheduler states
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load discriminator states
    for i, gan in enumerate(gans):
        accelerator.unwrap_model(gan.discriminator).load_state_dict(
            checkpoint[f'discriminator_{i}_state_dict']
        )
        gan.discriminator_opt.load_state_dict(checkpoint[f'discriminator_opt_{i}_state_dict'])
        gan.discriminator_scheduler.load_state_dict(checkpoint[f'discriminator_scheduler_{i}_state_dict'])

    # Load early stopper state if present
    if early_stopper is not None and 'early_stopper' in checkpoint:
        early_stopper.counter = checkpoint['early_stopper']['counter']
        early_stopper.opt_val = checkpoint['early_stopper']['opt_val']

    epoch = checkpoint['epoch']
    best_score = checkpoint.get('best_score', None)

    print(f"Resumed from epoch {epoch}, best_score: {best_score}")
    return epoch + 1, best_score  # Return next epoch to train
