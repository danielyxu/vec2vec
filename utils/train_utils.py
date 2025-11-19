import torch
import torch.nn.functional as F


rmse = lambda x, y: ((x - y) ** 2).sum(dim=1).sqrt().mean()

def rec_loss_fn(ins, recons, logger, prefix=""):
    assert ins.keys() == recons.keys()
    loss = None
    for flag, emb in ins.items():
        recons_loss = 1 - F.cosine_similarity(emb, recons[flag], dim=1).mean()
        logger.logkv(f"{prefix}{flag}_recons_rmse", rmse(emb, recons[flag]))
        logger.logkv(f"{prefix}{flag}_recons_cos", recons_loss)
        if loss is None:
            loss = recons_loss
        else:
            loss += recons_loss
    return loss / len(ins)


# def rec_margin_loss_fn(ins, recons, logger, prefix="", margin: float = 0.1):
#     """Penalizes embeddings from being more than `margin` similarity away from at least
#     one embedding."""
#     assert ins.keys() == recons.keys()
#     loss = None
#     for flag, emb in ins.items():
#         A = emb / emb.norm(dim=1, p=2, keepdim=True)
#         B = recons[flag] / recons[flag].norm(dim=1, p=2, keepdim=True)
#         # B = B.mean(dim=1, keepdim=True)

#         cos_distances = 1 - F.cosine_similarity(A, B, dim=1)
#         recons_loss_cos = cos_distances.mean()
#         margin_loss = (cos_distances - margin).clamp(min=0).mean()
#         logger.logkv(f"{prefix}{flag}_recons_rmse", rmse(emb, recons[flag]))
#         logger.logkv(f"{prefix}{flag}_recons_cos", recons_loss_cos)
#         if loss is None:
#             loss = margin_loss
#         else:
#             loss += margin_loss
#     return loss / len(ins)

def uni_loss_fn(emb, trans, src_emb, tgt_emb, logger):
    uni_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_rmse", rmse(emb, trans))
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_cos", uni_loss)
    return uni_loss


def trans_loss_fn(ins, translations, logger, prefix=""):
    assert ins.keys() == translations.keys()
    loss = None
    for target_flag, emb in ins.items():
        for flag, trans in translations[target_flag].items():
            trans_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_rmse", rmse(emb, trans))
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_cos", trans_loss)
            
            if loss is None:
                loss = trans_loss
            else:
                loss += trans_loss

    return (loss / len(ins))


def contrastive_loss_fn(ins, translations, logger) -> torch.Tensor:
# TODO: Think about this + test.
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
            out_sims_reflected = A @ B.T
            contrastive_loss = torch.nn.functional.cross_entropy(
                out_sims_reflected * 50,
                torch.arange(in_sims.shape[0], device=in_sims.device)
            )
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_contrastive", contrastive_loss)

            if loss is None:
                loss = contrastive_loss
            else:
                loss += contrastive_loss
            count += 1
    return loss / count

def vsp_loss_fn(ins, translations, logger, top_k=None) -> torch.Tensor:
    """
    Vector Set Preservation loss function.

    Args:
        ins: Input embeddings dictionary
        translations: Translated embeddings dictionary
        logger: Logger for metrics
        top_k: If specified, only average the top-k pairs with highest errors
               instead of averaging all pairs. This focuses training on the
               hardest examples (hard negative mining).

    Returns:
        VSP loss tensor
    """
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

            # Compute absolute differences
            vsp_diff = (in_sims - out_sims).abs()
            vsp_diff_reflected = (in_sims - out_sims_reflected).abs()

            if top_k is not None and top_k > 0:
                # Flatten and select top-k highest errors
                vsp_diff_flat = vsp_diff.flatten()
                vsp_diff_reflected_flat = vsp_diff_reflected.flatten()

                # Clamp top_k to not exceed total number of elements
                k = min(top_k, vsp_diff_flat.numel())

                # Get top-k values
                vsp_topk, _ = torch.topk(vsp_diff_flat, k)
                vsp_reflected_topk, _ = torch.topk(vsp_diff_reflected_flat, k)

                vsp_loss = vsp_topk.mean()
                vsp_loss_reflected = vsp_reflected_topk.mean()
            else:
                # Original behavior: average all pairs
                vsp_loss = vsp_diff.mean()
                vsp_loss_reflected = vsp_diff_reflected.mean()

            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_vsp", vsp_loss)
                logger.logkv(f"{in_name}_{out_name}_vsp_reflected", vsp_loss_reflected)

            if loss is None:
                loss = vsp_loss + vsp_loss_reflected
            else:
                loss += vsp_loss + vsp_loss_reflected
            count += 1
    return loss / count


def get_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.detach() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm