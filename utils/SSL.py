import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import torch
from models.model_set_mil import *
import numpy as np


def minmax_norm(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    min, max = float(arr.min()), float(arr.max())
    if max - min < eps:

        return np.zeros_like(arr, dtype=float)
    res = (arr - min) / (max - min)


    return res  # --- IGNORE ---


def c_index(times, events, preds):
    times = np.asarray(times); events = np.asarray(events); preds = np.asarray(preds)
    n = len(times)
    num, den = 0.0, 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (times[i] < times[j] and events[i] == 1):
                den += 1
                if preds[i] < preds[j]:
                    num += 1
                elif preds[i] == preds[j]:
                    num += 0.5
    return num / den if den > 0 else np.nan



def cosine_corr(x, y, mu, std, eps=1e-6):
    dev = y.device
    dt  = y.dtype

    x = x.to(device=dev, dtype=dt)
    y = y.to(device=dev, dtype=dt)
    mu  = torch.tensor(mu, device=dev, dtype=dt)
    std = torch.tensor(std, device=dev, dtype=dt).clamp(min=eps)
    xz = ((x.squeeze(0) - mu) / std)
    yz = ((y.squeeze(0) - mu) / std)
    xz = xz / (xz.norm() + eps); yz = yz / (yz.norm() + eps)
    cos = torch.clamp((xz * yz).sum(), -1.0, 1.0).item()
    return 1.0 - (np.arccos(cos) / np.pi)



def calculate_feature_similarity_vector(feat1, feat2, method='corr',
                                        mu=None, std=None, Sigma_inv=None):
    if method == 'corr':
        assert mu is not None and std is not None
        return cosine_corr(feat1, feat2, mu, std)

    else:
        raise ValueError(f'Unknown feature similarity {method}')



@torch.no_grad()
def extract_bank(model, train_loader, device='cuda'):
    model.eval()
    feats, attns, times, cens, paths, hazards_list = [], [], [], [], [], []
    for (img, _, t, c, p) in tqdm(train_loader, desc='Extract train bank'):
        if torch.equal(img, torch.ones(1)):
            continue
        img = img.to(device)
        # 之前只取 x, attn，这里把 hazards 也拿出来
        hazards, _, _, x, attn, _ = model(x_path=img)
        if attn is None:
            H, T = 4, x.shape[1]
            N = 64
            attn = torch.ones(H, T, N, device='cpu')
        feats.append(x.cpu())
        attns.append(attn.cpu())

        hazards_list.append(hazards.view(-1).detach().cpu().numpy().astype(float))
        times.extend(t.numpy() if isinstance(t, torch.Tensor) else t)
        cens.extend(c.cpu().numpy() if isinstance(c, torch.Tensor) else c)
        paths.extend(p)
    return feats, attns, np.array(times, dtype=float), np.array(cens, dtype=int), paths, hazards_list



def fit_feature_stats(train_features):
    X = torch.cat([f.squeeze(0).unsqueeze(0) for f in train_features], dim=0).cpu().numpy()
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(X)-1, 1)
    lam = 1e-2 * np.trace(cov) / cov.shape[0]
    Sigma = cov + lam * np.eye(cov.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)
    return mu.astype(np.float32), std.astype(np.float32), Sigma_inv.astype(np.float32)


def combine_bin_censor_to_label(bin_value, censor_value):
    """
    (bin:(0-3), censorship (0-1)):
    (0, 0) : 0
    (0, 1) : 1
    (1, 0) : 2
    (1, 1) : 3
    (2, 0) : 4
    (2, 1) : 5
    (3, 0) : 6
    (3, 1) : 7
    """
    return int(bin_value * 2 + censor_value)



@torch.no_grad()
def predict_bins_one_unlabeled_simple(
    x_query, attn_query,
    train_features, train_attentions,train_combined_labels,
    train_hazards,              
    query_hazards,             
    mu=None, std=None, Sigma_inv=None,
    k=10,                       
    use_topk=True,

):


    xr = x_query.cpu()
    ar = attn_query

    sf_list = []
    sa_list = []
    for xf, af in zip(train_features, train_attentions):

        sf = calculate_feature_similarity_vector(xr, xf, method='corr', mu=mu, std=std, Sigma_inv=Sigma_inv)
        _, sa = attention_kl_similarity_from_stats(ar, af)
        sf_list.append(float(sf))
        sa_list.append(float(sa))

    sf_list = np.array(sf_list, dtype=float)
    sa_list = np.array(sa_list, dtype=float)


    sa_norm = minmax_norm(sa_list)   #[num_train]
    sf_norm = minmax_norm(sf_list)

    s = 0.5 
    sims =  s*sf_norm +(1-s) * sa_norm  #  sf sa

    if use_topk:
        selected_idx = []
        num_classes = 8

        labels_all = np.array(train_combined_labels)
        for c in range(num_classes):
            cls_mask = (labels_all == c)
            if np.sum(cls_mask) == 0:
                continue

            cls_idx = np.where(cls_mask)[0]
            cls_sims = sims[cls_idx]
            cls_order = cls_idx[np.argsort(cls_sims)[::-1]]
            k_in_cls = min(k, len(cls_order))
            selected_idx.extend(cls_order[:k_in_cls])
        order = np.array(selected_idx)
    else:
        order = np.argsort(sims)[::-1]
        order = order[:k]

    used_k = len(order)

    contrib_sum = np.zeros(5, dtype=float)
    contrib_cnt = np.zeros(5, dtype=float)

    qh = np.asarray(query_hazards, dtype=float).reshape(-1)   
    assert qh.shape[0] == 4, f"query_hazards=4,In fact={qh.shape}"

    for idx in order:
        s = float(sims[idx])
        if s <= 0:
            continue
        lab = int(train_combined_labels[idx])

        if lab in (0, 2, 4, 6):
            b = lab // 2  # 0..3
            vec = np.zeros(5, dtype=float); vec[b] = 1.0 * s  # vec[b] = 1.0 * s
            contrib_sum += vec
            contrib_cnt += (vec > 0).astype(float)

        elif lab in (1, 3):
            b = (lab + 1) // 2   # b=1,2
            mask = np.zeros_like(qh); mask[b:] = 1.0
            qh = train_hazards[idx]  
            h_masked = qh * mask             


            if np.any(h_masked > 0):
                max_idx = np.argmax(h_masked)  
                h_masked = np.zeros_like(qh)  
                h_masked[max_idx] = 1.0       
            
            vec = np.zeros(5, dtype=float)
            vec[:4] = h_masked * s           
            contrib_sum += vec
            contrib_cnt += (vec > 0).astype(float)

        elif lab == 5:
            vec = np.zeros(5, dtype=float); vec[3] = 1.0 * s
            contrib_sum += vec
            contrib_cnt += (vec > 0).astype(float)

        elif lab == 7:
            vec = np.zeros(5, dtype=float); vec[4] = 1.0 * s
            contrib_sum += vec
            contrib_cnt += (vec > 0).astype(float)

        else:
            continue


    prob5 = np.zeros(5, dtype=float)
    nz = contrib_cnt > 0
    prob5[nz] = contrib_sum[nz] / contrib_cnt[nz]

    conf = float(prob5.max())
    dbg = dict(mode='simple_agg', used_k=used_k)

    return prob5, dbg, conf


@torch.no_grad()

def grid_validate_and_predict_bins(
    model, remove_loader, train_loader,args,
    k=5,
    device='cuda',
    bins=None,
    ema_bank=None,             
    ema_filter_paths=None      
):

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # 1) momery bank
    if (ema_bank is not None) and (len(ema_bank) > 0):
        train_features, train_attentions, train_times, train_cens, train_paths, train_hazards = \
            ema_bank.export_for_grid(filter_paths=ema_filter_paths)
    else:
        train_features, train_attentions, train_times, train_cens, train_paths, train_hazards = \
            extract_bank(model, train_loader, device=device)
    train_attn_stats = [attention_stats(a) for a in train_attentions]
    mu, std, Sigma_inv = fit_feature_stats(train_features)

    taus = np.asarray(bins, dtype=float); K = len(taus) - 1
    def time_to_bin(t):
        j = np.searchsorted(taus, t, side='right') - 1
        return int(np.clip(j, 0, K-1))
    train_bins = [time_to_bin(t) for t in train_times]
    train_combined_labels = [combine_bin_censor_to_label(bv, cv) for bv, cv in zip(train_bins, train_cens)]


    model.eval()
    rm_feats, rm_attns,   rm_paths = [], [], []
    rm_hazards = []  

    for (img_r, lab_r, t_r, c_r, p_r) in tqdm(remove_loader, desc='Extract remove set'):
        if torch.equal(img_r, torch.ones(1)):
            continue
        img_r = img_r.to(device)
        hazards, S, _, x_r, attn_r, _ = model(x_path=img_r)
        if attn_r is None:
            H, T = 4, x_r.shape[1]
            N = 64
            attn_r = torch.ones(H, T, N, device='cpu')


        rm_feats.append(x_r.cpu())
        rm_attns.append(attn_r.cpu())
        rm_paths.append(p_r[0])
        rm_hazards.append(hazards.view(-1).detach().cpu().numpy().astype(float))

    rm_attn_stats = [attention_stats(a) for a in rm_attns]
    preds_bin_out, probs5_list, pred_combined_labels = [], [], []
    pseudo_times, pred_censored_hat_list = [], []
    pred_confidences = [] 

    last_width = float(bins[-1] - bins[-2]) if len(bins) >= 2 else 6.0
    extra_delta = max(1.0, 0.5 * last_width)

    for x_r, a_r,  sp, qh in tqdm(
        list(zip(rm_feats, rm_attn_stats,  rm_paths, rm_hazards)),
        desc='Predict (simple)'
    ):
        prob5, dbg, conf = predict_bins_one_unlabeled_simple(
                                                            x_query=x_r, attn_query=a_r,
                                                            train_features=train_features, train_attentions=train_attn_stats,
                                                            train_combined_labels=train_combined_labels,
                                                            train_hazards=train_hazards,          
                                                            query_hazards=qh,                      
                                                            mu=mu, std=std, Sigma_inv=Sigma_inv,
                                                            k=k, use_topk=True,
                                                        )

        pred_confidences.append(float(conf))

        idx = int(np.argmax(prob5))
        if idx == 4:
            pred_bin_out = K - 1
            pred_combined = 7
            pred_cen_hat  = 1
            pseudo_t = float(bins[-2] + extra_delta)
        else:
            pred_bin_out = idx
            pred_combined = 2*idx + 0
            pred_cen_hat  = 0
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            pseudo_t = float(bin_centers[idx])

        preds_bin_out.append(int(pred_bin_out))
        probs5_list.append(json.dumps([float(x) for x in prob5.tolist()]))
        pred_combined_labels.append(int(pred_combined))
        pred_censored_hat_list.append(int(pred_cen_hat))
        pseudo_times.append(pseudo_t)

    df_pred = pd.DataFrame({
        'path': rm_paths,
        'slide_id': rm_paths,
        'pred_bin_out': preds_bin_out,
        'pred_censored_hat': pred_censored_hat_list,
        'pseudo_time': pseudo_times,     
        'pred_confidence': pred_confidences,  #

    })

    return df_pred 





def kl_gaussian_1d(mu_p, std_p, mu_q, std_q, eps=1e-6):
    """
    KL( N(mu_p, std_p^2) || N(mu_q, std_q^2) ), 
    shape: mu_p, std_p, mu_q, std_q 都是 [32]
    """
    std_p = torch.clamp(std_p, min=eps)
    std_q = torch.clamp(std_q, min=eps)

    var_p = std_p ** 2
    var_q = std_q ** 2

    return torch.log(std_q / std_p) + (var_p + (mu_p - mu_q) ** 2) / (2 * var_q) - 0.5



def attention_stats(attn):
    attn = attn.float().cpu()

    if attn.dim() != 3:
        raise ValueError(f"attn shape error: {attn.shape}")

    B, C, N = attn.shape
    attn = attn.reshape(B * C, N)

    mu = attn.mean(dim=1)
    std = attn.std(dim=1, unbiased=False)

    return mu, std


def attention_kl_similarity_from_stats(stat_r, stat_f, eps=1e-6):
    mu_r, std_r = stat_r
    mu_f, std_f = stat_f

    std_r = torch.clamp(std_r, min=eps)
    std_f = torch.clamp(std_f, min=eps)

    kl_f_r = kl_gaussian_1d(mu_f, std_f, mu_r, std_r)

    mean_kl = kl_f_r.mean()
    sim = torch.exp(-mean_kl / 10.0)

    return mean_kl, sim


if __name__ == "__main__":
    for i in range(5):
        print(f's_{i}')
