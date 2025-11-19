import torch
import torch.nn.functional as F
from typing import Optional


rmse = lambda x, y: ((x - y) ** 2).sum(dim=1).sqrt().mean()


def _compute_triplet_angles(vectors: torch.Tensor, num_triplets: int = 256) -> torch.Tensor:
    """
    Compute angles at the middle vertex of randomly sampled triplets.

    For triplet (i, j, k), computes angle at vertex j using vectors (v_i - v_j) and (v_k - v_j).

    Args:
        vectors: (batch_size, dim) tensor of vectors
        num_triplets: number of triplets to sample

    Returns:
        (num_triplets,) tensor of cosine of angles
    """
    batch_size = vectors.shape[0]
    if batch_size < 3:
        return torch.zeros(1, device=vectors.device)

    # Sample triplet indices
    indices = torch.randint(0, batch_size, (num_triplets, 3), device=vectors.device)

    # Ensure distinct indices within each triplet
    # Simple rejection: just resample if duplicates (rare for large batch)
    i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]

    v_i = vectors[i_idx]  # (num_triplets, dim)
    v_j = vectors[j_idx]
    v_k = vectors[k_idx]

    # Vectors from j to i and j to k
    vec_ji = v_i - v_j  # (num_triplets, dim)
    vec_jk = v_k - v_j

    # Compute cosine of angle at j
    EPS = 1e-8
    norm_ji = vec_ji.norm(dim=1, keepdim=True).clamp(min=EPS)
    norm_jk = vec_jk.norm(dim=1, keepdim=True).clamp(min=EPS)

    cos_angle = (vec_ji * vec_jk).sum(dim=1) / (norm_ji.squeeze() * norm_jk.squeeze())

    return cos_angle


def _compute_tetrahedron_volumes(vectors: torch.Tensor, num_tetrahedra: int = 128) -> torch.Tensor:
    """
    Compute volumes of randomly sampled tetrahedra formed by 4 points.

    Volume = |det([v2-v1, v3-v1, v4-v1])| / 6

    Args:
        vectors: (batch_size, dim) tensor of vectors
        num_tetrahedra: number of tetrahedra to sample

    Returns:
        (num_tetrahedra,) tensor of volumes
    """
    batch_size, dim = vectors.shape
    if batch_size < 4:
        return torch.zeros(1, device=vectors.device)

    # Sample quadruplet indices
    indices = torch.randint(0, batch_size, (num_tetrahedra, 4), device=vectors.device)

    v1 = vectors[indices[:, 0]]  # (num_tetrahedra, dim)
    v2 = vectors[indices[:, 1]]
    v3 = vectors[indices[:, 2]]
    v4 = vectors[indices[:, 3]]

    # Edge vectors from v1
    e1 = v2 - v1  # (num_tetrahedra, dim)
    e2 = v3 - v1
    e3 = v4 - v1

    # For high-dimensional vectors, we compute a "generalized volume"
    # using the Gram determinant: sqrt(det(G)) where G_ij = e_i . e_j
    # This gives the 3D volume of the parallelepiped spanned by projections

    # Gram matrix elements
    g11 = (e1 * e1).sum(dim=1)
    g12 = (e1 * e2).sum(dim=1)
    g13 = (e1 * e3).sum(dim=1)
    g22 = (e2 * e2).sum(dim=1)
    g23 = (e2 * e3).sum(dim=1)
    g33 = (e3 * e3).sum(dim=1)

    # Determinant of 3x3 Gram matrix
    # det(G) = g11*(g22*g33 - g23^2) - g12*(g12*g33 - g23*g13) + g13*(g12*g23 - g22*g13)
    det_G = (
        g11 * (g22 * g33 - g23 * g23) -
        g12 * (g12 * g33 - g23 * g13) +
        g13 * (g12 * g23 - g22 * g13)
    )

    # Volume = sqrt(det(G)) / 6 (for tetrahedron, it's 1/6 of parallelepiped)
    # Use abs to handle numerical issues
    volume = torch.sqrt(det_G.abs().clamp(min=1e-12)) / 6.0

    return volume


def vsp_triplet_angle_loss(
    source_vectors: torch.Tensor,
    translated_vectors: torch.Tensor,
    num_triplets: int = 256
) -> torch.Tensor:
    """
    Compute triplet angle preservation loss.

    Ensures that angles between triplets of vectors are preserved after translation.
    """
    source_angles = _compute_triplet_angles(source_vectors, num_triplets)
    translated_angles = _compute_triplet_angles(translated_vectors, num_triplets)

    # MAE between angles
    loss = (source_angles - translated_angles).abs().mean()
    return loss


def vsp_tetrahedron_loss(
    source_vectors: torch.Tensor,
    translated_vectors: torch.Tensor,
    num_tetrahedra: int = 128
) -> torch.Tensor:
    """
    Compute tetrahedron volume preservation loss.

    Ensures that volumes of tetrahedra formed by 4 points remain consistent.
    """
    source_volumes = _compute_tetrahedron_volumes(source_vectors, num_tetrahedra)
    translated_volumes = _compute_tetrahedron_volumes(translated_vectors, num_tetrahedra)

    # Normalize volumes to make loss scale-invariant
    EPS = 1e-8
    source_norm = source_volumes / (source_volumes.mean() + EPS)
    translated_norm = translated_volumes / (translated_volumes.mean() + EPS)

    # MAE between normalized volumes
    loss = (source_norm - translated_norm).abs().mean()
    return loss

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

def vsp_loss_fn(
    ins,
    translations,
    logger,
    vsp_mode: str = 'pairwise',
    vsp_triplet_coeff: float = 1.0,
    vsp_tetra_coeff: float = 1.0,
    vsp_num_triplets: int = 256,
    vsp_num_tetrahedra: int = 128
) -> torch.Tensor:
    """
    Vector Space Preservation loss with multiple modes.

    Args:
        ins: Dictionary of input embeddings
        translations: Dictionary of translated embeddings
        logger: Logger for metrics
        vsp_mode: One of 'pairwise', 'triplet_angle', 'tetrahedron', 'combined'
            - 'pairwise': Original VSP - preserves pairwise dot product similarities
            - 'triplet_angle': Preserves angles between triplets of vectors
            - 'tetrahedron': Preserves volumes of tetrahedra formed by 4 points
            - 'combined': All three modes combined
        vsp_triplet_coeff: Weight for triplet angle loss (used in 'triplet_angle' and 'combined' modes)
        vsp_tetra_coeff: Weight for tetrahedron volume loss (used in 'tetrahedron' and 'combined' modes)
        vsp_num_triplets: Number of triplets to sample per batch
        vsp_num_tetrahedra: Number of tetrahedra to sample per batch

    Returns:
        Combined VSP loss tensor
    """
    loss = None
    EPS = 1e-10
    count = 0

    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B_norm = B / (B.norm(dim=1, keepdim=True) + EPS)
            A = translations[out_name][in_name]
            A_norm = A / (A.norm(dim=1, keepdim=True) + EPS)

            pair_loss = torch.tensor(0.0, device=B.device)
            triplet_loss = torch.tensor(0.0, device=B.device)
            tetra_loss = torch.tensor(0.0, device=B.device)

            # Pairwise similarity preservation (original VSP)
            if vsp_mode in ['pairwise', 'combined']:
                in_sims = B_norm @ B_norm.T
                out_sims = A_norm @ A_norm.T
                out_sims_reflected = A_norm @ B_norm.T
                pair_loss = (in_sims - out_sims).abs().mean()
                pair_loss_reflected = (in_sims - out_sims_reflected).abs().mean()
                pair_loss = pair_loss + pair_loss_reflected

                if logger is not None:
                    logger.logkv(f"{in_name}_{out_name}_vsp_pairwise", pair_loss)
                    logger.logkv(f"{in_name}_{out_name}_vsp_reflected", pair_loss_reflected)

            # Triplet angle preservation
            if vsp_mode in ['triplet_angle', 'combined']:
                triplet_loss = vsp_triplet_angle_loss(B_norm, A_norm, vsp_num_triplets)
                # Also compute reflected version
                triplet_loss_reflected = vsp_triplet_angle_loss(B_norm, A_norm, vsp_num_triplets)
                triplet_loss = triplet_loss + triplet_loss_reflected

                if logger is not None:
                    logger.logkv(f"{in_name}_{out_name}_vsp_triplet", triplet_loss)

            # Tetrahedron volume preservation
            if vsp_mode in ['tetrahedron', 'combined']:
                tetra_loss = vsp_tetrahedron_loss(B_norm, A_norm, vsp_num_tetrahedra)
                # Also compute reflected version
                tetra_loss_reflected = vsp_tetrahedron_loss(B_norm, A_norm, vsp_num_tetrahedra)
                tetra_loss = tetra_loss + tetra_loss_reflected

                if logger is not None:
                    logger.logkv(f"{in_name}_{out_name}_vsp_tetra", tetra_loss)

            # Combine losses based on mode
            if vsp_mode == 'pairwise':
                current_loss = pair_loss
            elif vsp_mode == 'triplet_angle':
                current_loss = triplet_loss * vsp_triplet_coeff
            elif vsp_mode == 'tetrahedron':
                current_loss = tetra_loss * vsp_tetra_coeff
            elif vsp_mode == 'combined':
                current_loss = (
                    pair_loss +
                    triplet_loss * vsp_triplet_coeff +
                    tetra_loss * vsp_tetra_coeff
                )
            else:
                raise ValueError(f"Unknown vsp_mode: {vsp_mode}. "
                                f"Choose from: 'pairwise', 'triplet_angle', 'tetrahedron', 'combined'")

            # Log total VSP loss for this pair
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_vsp", current_loss)

            if loss is None:
                loss = current_loss
            else:
                loss += current_loss
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