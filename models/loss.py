import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- New disentangling losses (drop-in replacement for margin/cos loss) ----------
def gram_orth_loss(feats: torch.Tensor) -> torch.Tensor:
    """
    feats: [K, d]  -- x_local after compression (already from model)
    Encourage pairwise orthogonality in compressed space.
    L = ||G - I||_F^2 (off-diagonal normalized)
    """
    K = feats.size(0)
    if K <= 1:
        return torch.tensor(0., device=feats.device)
    f = F.normalize(feats, dim=1)                      # [K, d]
    G = torch.matmul(f, f.t())                         # [K, K]
    off = G - torch.eye(K, device=feats.device)
    # 只统计非对角，避免 I 的主对角影响
    return (off.pow(2).sum() / (K * (K - 1)))

def energy_balance_loss(feats: torch.Tensor) -> torch.Tensor:
    """
    Keep prototype energies balanced to avoid collapse.
    L = Var(||f_i||_2) over i
    """
    norms = feats.norm(dim=1)                          # [K]
    return norms.var(unbiased=False)

def classifier_group_orth_loss(model) -> torch.Tensor:
    """
    Make per-prototype classifier weights orthogonal (decision-path disentanglement).
    We reshape classifier_local weight to [C, K, d_local], average over classes -> [K, d_local],
    then penalize off-diagonal cosine sims as in Gram orthogonalization.
    """
    W = model.classifier_local.weight                  # [C, K*d_local]
    K = model.num_prototype - 1
    d_local = model.c_local
    if W.dim() != 2 or W.size(1) != K * d_local:
        return torch.tensor(0., device=W.device)
    Wg = W.view(W.size(0), K, d_local)                 # [C, K, d_local]
    Wp = Wg.mean(dim=0)                                # [K, d_local]
    Wp = F.normalize(Wp, dim=1)
    G = Wp @ Wp.t()                                    # [K, K]
    off = G - torch.eye(K, device=W.device)
    return (off.pow(2).sum() / (K * (K - 1)))

# 建议的权重（起点）：可通过验证集微调
def compute_disentangle_loss(model, x_local,
                            lambda_orth=1.0,
                            lambda_bal=0.2,
                            lambda_wdec=0.5):
    L_orth = gram_orth_loss(x_local)
    L_bal  = energy_balance_loss(x_local)
    L_wdec = classifier_group_orth_loss(model)
    return lambda_orth * L_orth + lambda_bal * L_bal + lambda_wdec * L_wdec, \
        {'L_orth': L_orth.item(), 'L_bal': L_bal.item(), 'L_wdec': L_wdec.item()}
