import pickle
import random

import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
# import torch_geometric
# from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def proto_contrastive_loss_patch(
    prototypes: torch.Tensor,   # [K, D] or [1, K, D]
    patches: torch.Tensor,      # [N, D] or [1, N, D]
    temperature: float = 0.2,
    p_threshold: float = None,        # default: 1/K (strictly >)

):
    
    # ---- 0) squeeze batch=1 ----
    if prototypes.dim() == 3:
        assert prototypes.size(0) == 1 and patches.size(0) == 1 and attn.size(0) == 1
        prototypes = prototypes[0]  # [K, D]
        patches = patches[0]        # [N, D]
        attn = attn[0]              # [K, N]

    K, D = prototypes.shape
    N, D2 = patches.shape

    similarity_matrix = F.cosine_similarity(
        prototypes.unsqueeze(1),  # [K, 1, D]
        patches.unsqueeze(0),         # [1, N, D]
        dim=-1
    )  # [K, N]
    attn = similarity_matrix

    assert D == D2
    assert attn.shape == (K, N)

    if p_threshold is None:
        p_threshold = 1.0 / K  

    # ---- 1) softmax over prototypes for each patch ----
    # For each patch i: sum_k attn_soft[k,i] = 1
    attn_soft = F.softmax(attn, dim=0)  # [K, N]

    # ---- 2) precompute logits for ALL patches to ALL protos ----
    proto_norm = F.normalize(prototypes, p=2, dim=1)   # [K, D]
    patch_norm = F.normalize(patches, p=2, dim=1)      # [N, D]
    logits_all = (patch_norm @ proto_norm.t()) / temperature  # [N, K]

    # ---- 3) for each proto k: 
    all_idx = []
    all_labels = []

    for k in range(K):
        eligible = (attn_soft[k] > p_threshold).nonzero(as_tuple=False).squeeze(1)  # indices in [0..N-1]
        if eligible.numel() == 0:
            continue
        idx_k = eligible  
        all_idx.append(idx_k)
        all_labels.append(torch.full((idx_k.size(0),), k, device=idx_k.device, dtype=torch.long))

    if len(all_idx) == 0:
        return torch.tensor(0.0, device=patches.device)

    idx = torch.cat(all_idx, dim=0)          # [M]
    labels = torch.cat(all_labels, dim=0)    # [M]

    logits = logits_all.index_select(0, idx) # [M, K]

    # ---- 4) InfoNCE = CE over prototypes ----
    loss = F.cross_entropy(logits, labels)
    return loss

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def collate_MIL_survival_training(batch):
    # for  (data_WSI, label, event_time, c, path) in batch:
    #     print(path)
    img = torch.cat([item[0] for item in batch], dim=0)

    # feat_length = img.size(1)
    # p = random.uniform(0.05, 0.2)
    # mask_0 = np.random.choice(feat_length, int(p*feat_length))
    # mask_1 = torch.ones((1, feat_length))
    # mask_1[:, mask_0] = 0
    # img = img * mask_1

    # bag_length = img.size(0)
    # if not bag_length == 1:
    #     p = random.uniform(0.8, 1)
    #     mask_0 = np.random.choice(bag_length, int(p * bag_length))
    #     img = img[mask_0, :]

    label = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    path = [item[4] for item in batch]
    return [img, label, event_time, c, path]


def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    path = [item[4] for item in batch]
    return [img, label, event_time, c, path]


def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, cluster_ids, label, event_time, c]


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 


def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    collate = collate_MIL_survival
    collate_training = collate_MIL_survival_training

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weights, len(weights)), collate_fn = collate_training, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler=RandomSampler(split_dataset), collate_fn = collate_training, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler=SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

# =========================
# EMA Memory Bank (attn + feat)
# =========================


import torch
import numpy as np

class EMABank:

    def __init__(
        self,
        momentum: float = 0.9,
        eps: float = 1e-8,
        store_on_cpu: bool = False,
        attn_norm_dim: int = -1,   # attn 做归一化的维度（默认最后一维）
        dtype_feat: torch.dtype = torch.float32,
        dtype_attn: torch.dtype = torch.float32,
    ):
        self.m = float(momentum)
        self.eps = float(eps)
        self.store_on_cpu = bool(store_on_cpu)
        self.attn_norm_dim = int(attn_norm_dim)
        self.dtype_feat = dtype_feat
        self.dtype_attn = dtype_attn

        self._store = {}  # path -> dict(feat, attn, hazards, time, cens)

    def __len__(self):
        return len(self._store)

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _to_list(v):
        """tensor/list/scalar -> python list"""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().view(-1).tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    @staticmethod
    def _is_batched_tensor(x: torch.Tensor, B: int) -> bool:
        """是否把第0维当 batch：仅当 x.dim>=1 且 x.size(0)==B"""
        return isinstance(x, torch.Tensor) and x.dim() >= 1 and x.size(0) == B

    def _split_to_list(self, x, B: int, name: str):
        """
        把输入 x（tensor/list/None）转换成长度为 B 的 list（或长度1的 list）。
        关键：只有当第0维确实是 batch（size(0)==B）才按 batch 切；
             否则视为单样本，不切（例如 attn 是 [4,8,N]）。
        """
        if x is None:
            return [None] * B

        if isinstance(x, (list, tuple)):
            # 直接作为 list（长度不够时截断/补 None）
            xl = list(x)
            if len(xl) >= B:
                return xl[:B]
            return xl + [None] * (B - len(xl))

        if isinstance(x, torch.Tensor):
            if self._is_batched_tensor(x, B):
                return [x[i] for i in range(B)]
            else:
                # 单样本（无 batch 维）
                return [x] + [None] * (B - 1)

        # 其它类型当作单样本
        return [x] + [None] * (B - 1)

    def _canon_feat(self, f):
        """feat 张量：detach + dtype + (cpu可选)"""
        if not isinstance(f, torch.Tensor):
            f = torch.as_tensor(f)
        f = f.detach().to(dtype=self.dtype_feat)
        if self.store_on_cpu:
            f = f.cpu()
        return f

    def _canon_attn(self, a, feat_fallback=None):
        """
        attn 张量：detach + dtype + (cpu可选) + 非负 + 归一化
        如果 a 是 None：用占位 ones，尽量与旧逻辑兼容
        """
        if a is None:
            # 占位：默认 [4, T, 64]（你之前的逻辑）
            T = 1
            if isinstance(feat_fallback, torch.Tensor) and feat_fallback.dim() >= 2:
                T = feat_fallback.shape[1]
            a = torch.ones(4, T, 64, dtype=self.dtype_attn)

        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a)

        a = a.detach().to(dtype=self.dtype_attn)
        if self.store_on_cpu:
            a = a.cpu()

        # 非负 & 归一化（按 attn_norm_dim）
        a = torch.clamp(a, min=self.eps)
        denom = a.sum(dim=self.attn_norm_dim, keepdim=True).clamp(min=self.eps)
        a = a / denom
        return a

    @staticmethod
    def _canon_hazards(h):
        """hazards：转 numpy float array"""
        if h is None:
            return None
        if isinstance(h, torch.Tensor):
            return h.detach().cpu().view(-1).numpy().astype(float)
        return np.asarray(h, dtype=float).reshape(-1)

    # -------------------------
    # main api
    # -------------------------
    @torch.no_grad()
    def update(
        self,
        paths,
        feats,
        attns,
        hazards,
        times,
        cens,
        momentum: float = None,

    ):
        """
        paths: list[str] or str
        feats: Tensor or list[Tensor]
        attns: Tensor or list[Tensor]  (关键：不要把 [4,8,N] 当 batch 切)
        hazards: Tensor or list / np
        times, cens: tensor/list/scalar
        """

        m = self.m if momentum is None else float(momentum)

        # ---- path list ----
        if isinstance(paths, (tuple, list)):
            path_list = list(paths)
        else:
            path_list = [paths]

        B = len(path_list)

        # ---- split other inputs by B (only if true batch dim) ----
        feat_list = self._split_to_list(feats, B, "feats")
        attn_list = self._split_to_list(attns, B, "attns")
        hz_list   = self._split_to_list(hazards, B, "hazards")

        t_list = self._to_list(times)
        c_list = self._to_list(cens)

        # t/c 长度不足时补齐（常见：times 是 np array 长度 B，c 是 tensor 长度 B）
        if len(t_list) < B:
            t_list = t_list + [t_list[-1]] * (B - len(t_list))
        if len(c_list) < B:
            c_list = c_list + [c_list[-1]] * (B - len(c_list))

        for i in range(B):
            p = path_list[i]

            f_raw = feat_list[i]
            a_raw = attn_list[i]
            h_raw = hz_list[i]

            # 有些 split_to_list 会补 None（比如你传了单样本但 B>1）
            if f_raw is None:
                continue  # 没特征就跳过

            f = self._canon_feat(f_raw)
            a = self._canon_attn(a_raw, feat_fallback=f)
            h = self._canon_hazards(h_raw)

            if p not in self._store:
                self._store[p] = dict(
                    feat=f.clone(),
                    attn=a.clone(),
                    hazards=h.copy() if h is not None else None,
                    time=float(t_list[i]),
                    cens=int(c_list[i]),
                )
            else:
                old = self._store[p]

                # ---- feat EMA（形状必须一致，否则直接覆盖） ----
                if isinstance(old["feat"], torch.Tensor) and old["feat"].shape == f.shape:
                    old["feat"] = old["feat"] * m + f * (1.0 - m)
                else:
                    old["feat"] = f.clone()

                # ---- attn EMA（形状必须一致，否则直接覆盖） ----
                if isinstance(old["attn"], torch.Tensor) and old["attn"].shape == a.shape:
                    new_a = old["attn"] * m + a * (1.0 - m)
                else:
                    new_a = a.clone()

                # 归一化保持分布
                new_a = torch.clamp(new_a, min=self.eps)
                denom = new_a.sum(dim=self.attn_norm_dim, keepdim=True).clamp(min=self.eps)
                new_a = new_a / denom

                old["attn"] = new_a
                old["hazards"] = h.copy() if h is not None else None
                old["time"] = float(t_list[i])
                old["cens"] = int(c_list[i])

    def export_for_grid(self, filter_paths=None):
        """
        输出格式对齐你原 extract_bank 的返回：
          feats(list[Tensor]), attns(list[Tensor]),
          times(np.ndarray), cens(np.ndarray), paths(list[str]),
          hazards_list(list[np.ndarray or None])
        """
        if filter_paths is None:
            keys = list(self._store.keys())
        else:
            fset = set(filter_paths)
            keys = [k for k in self._store.keys() if k in fset]

        keys.sort()

        feats, attns, times, cens, hazards_list = [], [], [], [], []
        for k in keys:
            e = self._store[k]
            feats.append(e["feat"])
            attns.append(e["attn"])
            times.append(float(e["time"]))
            cens.append(int(e["cens"]))
            hazards_list.append(e["hazards"])

        return feats, attns, np.asarray(times, dtype=float), np.asarray(cens, dtype=int), keys, hazards_list






def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    pdb.set_trace()
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    # breakpoint()
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

def pamt_fixed_prototype_loss(attn, x_patch, top_h=10, temperature=0.1):
    """
    修复注意力分布问题的版本
    """
    num_prototypes, num_patches = attn.shape[0]*attn.shape[1], attn.shape[2]
    attn_merged = attn.reshape(num_prototypes, num_patches)  # [20, 5740]
    
    # 问题：原始注意力值太小，导致softmax后分布平坦
    # 解决方案：对注意力值进行缩放
    attn_scaled = attn_merged * 10.0  # 放大10倍，让softmax更有区分度
    
    # 方法1：先对注意力进行softmax归一化
    attn_normalized = F.softmax(attn_scaled, dim=1)  # [20, 5740]
   # print(f"Fixed Normalized attn range: [{attn_normalized.min().item():.6f}, {attn_normalized.max().item():.6f}]")
    
    # 检查注意力分布的质量
    max_attention = attn_normalized.max(dim=1).values
  #  print(f"每个原型的最大注意力: min={max_attention.min().item():.6f}, max={max_attention.max().item():.6f}, mean={max_attention.mean().item():.6f}")
    
    # 如果最大注意力仍然太小，继续调整缩放因子
    if max_attention.mean() < 0.01:  # 如果平均最大注意力小于1%
        scaling_factor = 20.0  # 增大缩放因子
        attn_scaled = attn_merged * scaling_factor
        attn_normalized = F.softmax(attn_scaled, dim=1)
   #     print(f"重新缩放后 attn range: [{attn_normalized.min().item():.6f}, {attn_normalized.max().item():.6f}]")
    
    # 得到原型特征表示
    prototype_features = torch.matmul(attn_normalized, x_patch)  # [20, 1024]
    
    # 对特征进行归一化
    prototype_features_norm = F.normalize(prototype_features, p=2, dim=1)
    x_patch_norm = F.normalize(x_patch, p=2, dim=1)
    
    # 计算余弦相似度
    S = torch.matmul(prototype_features_norm, x_patch_norm.T)  # [20, 5740]
    S = torch.clamp(S, -1.0 + 1e-8, 1.0 - 1e-8)
    
    # 后续计算保持不变...
    topk_values, topk_indices = torch.topk(S, k=top_h, dim=1)
    S_exp = torch.exp(S / temperature)
    denominators = S_exp.sum(dim=1, keepdim=True)
    
    mask = torch.zeros_like(S, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    
    S_prime = torch.zeros_like(S)
    S_prime[mask] = S_exp[mask] / denominators.repeat(1, num_patches)[mask]
    
    Y_h = torch.zeros_like(S)
    Y_h.scatter_(1, topk_indices, 1.0 / top_h)
    
    loss = - (Y_h * torch.log(S_prime + 1e-8)).sum(dim=1).mean()
    
    return loss, S, topk_values

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class SoftBCEHazardLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, hazards, soft_hazard_targets):
        h = torch.clamp(hazards, self.eps, 1.0 - self.eps)          # [B,K], 每段hazard
        t = torch.clamp(soft_hazard_targets, self.eps, 1.0 - self.eps)  # [B,K], 逐段hazard软标签
        # 逐段BCE：希望 h_j ≈ t_j
        loss = -(t * torch.log(h) + (1 - t) * torch.log(1 - h)).mean()
        return loss

    # 放在 utils/loss.py 或你现有 loss 定义处旁边
class SoftNLLSurvLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, hazards: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        hazards: [B, K] in (0,1)
        target_probs: [B, K], each row will be renormalized to sum=1
        """
        h = torch.clamp(hazards, self.eps, 1.0 - self.eps)        # 防止 0/1
        S = torch.cumprod(1.0 - h, dim=1)                         # [B,K]
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)  # S_{j-1}
        f = torch.clamp(S_prev * h, self.eps, 1.0)                # 事件概率

        p = target_probs
        p = p / (p.sum(dim=1, keepdim=True) + self.eps)           # 保证归一
        loss = -(p * torch.log(f)).sum(dim=1).mean()
        return loss
# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    #reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = S[j] >= S[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


def get_custom_exp_code(args):
    # New exp_code
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    param_code = ''

    param_code += args.model_type

    if args.resample > 0:
        param_code += '_resample'

    param_code += '_%s' % args.bag_loss

    param_code += '_a%s' % str(args.alpha_surv)

    if args.lr != 2e-4:
        param_code += '_lr%s' % format(args.lr, '.0e')

    if args.reg_type != 'None':
        param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    param_code += '_%s' % args.which_splits.split("_")[0]

    if args.batch_size != 1:
        param_code += '_b%s' % str(args.batch_size)

    if args.gc != 1:
        param_code += '_gc%s' % str(args.gc)

    args.exp_code = exp_code + '_' + param_code
    args.param_code = param_code

    return args