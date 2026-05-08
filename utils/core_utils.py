from argparse import Namespace
import os
from sksurv.metrics import concordance_index_censored
from importlib.machinery import SourceFileLoader
import torch
from models.loss import *
from utils.utils import *
from torch.optim import lr_scheduler
from datasets.dataset_generic import save_splits
from models.model_set_mil import *
from utils.SSL import grid_validate_and_predict_bins



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def _build_pseudo_from_grid(model, train_loader, remove_loader, bins, args,
                            k=5, agree_thr=0.6, hard_thr=0.6, device='cuda', epoch=None,ema_bank=None):

    df = grid_validate_and_predict_bins(model=model,remove_loader=remove_loader,train_loader=train_loader,args = args, k=k,device=device,bins=bins, ema_bank=ema_bank )
    
    records = []
    all_candidates = []

    for _, row in df.iterrows():
            all_candidates.append({
                'path': row['path'],
                'label': int(row['pred_bin_out']),       
                'event_time': float(row['pseudo_time']), 
                'c': int(row['pred_censored_hat']),      
                'hard_conf': float(row.get('pred_confidence'))
            })
    
    if not all_candidates:
        return records
    

    all_candidates.sort(key=lambda x: x['hard_conf'], reverse=True)
    

    total_candidates = len(all_candidates)
    
    if epoch is None:
        selected_count = sum(1 for candidate in all_candidates if candidate['hard_conf'] >= 0.96)
    else:
        if epoch < 5:
            select_ratio = 0.02* (epoch+1) 
            selected_count = max(1, int(total_candidates * select_ratio))
        else:
            select_ratio = 0.02*5
            selected_count = max(1, int(total_candidates * select_ratio))

    selected_candidates = all_candidates[:selected_count]
    

    for candidate in selected_candidates:
        records.append({
            'path': candidate['path'],
            'label': candidate['label'],
            'event_time': candidate['event_time'],
            'c': candidate['c']
        })
    

    return records


def train(datasets: tuple, cur: int, args: Namespace, bins_train=None):
    print('\nTraining Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, remove_split = datasets
    save_splits(datasets, ['train', 'val','remove'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("remove on {} samples".format(len(remove_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    reg_fn = None

    print('\nInit Model...', end=' ')

    model_dict = {'n_classes': args.n_classes}
    model = MIL_Attention_FC_surv(n_classes=4).cuda()

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    if args.semi_sup == True:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8) #0.8 
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # 0.5

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split,
                                    training=True,
                                    testing=args.testing,
                                    weighted=args.weighted_sample,
                                    mode=args.mode,
                                    batch_size=args.batch_size)

    val_loader = get_split_loader(val_split,
                                  testing=args.testing,
                                  mode=args.mode,
                                  batch_size=args.batch_size)
    remove_loader = get_split_loader(remove_split, 
                                     testing=False, 
                                     mode=args.mode, 
                                     batch_size=args.batch_size)
    
    base_train_ds = train_loader.dataset  
    pseudo_ds = None                      

    if args.semi_sup == True:
        ema_bank = EMABank(momentum=getattr(args, "bank_momentum", 0.9))
    else:
        ema_bank = None


    gt_path_set = set()   

    train_loss_list, c_index_list = [], []
    for epoch in range(args.max_epochs):
        train_loss_survival = train_loop_survival(args, epoch, model, train_loader, optimizer, scheduler, loss_fn, reg_fn, args.lambda_reg, args.gc,ema_bank=ema_bank, bank_momentum=getattr(args, "bank_momentum", 0.9), gt_path_set=gt_path_set)
        c_index = validate_survival(epoch, model, val_loader, loss_fn, reg_fn, args.lambda_reg)
        if epoch >= args.max_epochs - 4:
            train_loss_list.append(train_loss_survival)
            c_index_list.append(c_index)
        
        semi_epoch = 0
        if args.semi_sup ==True:
        
            if epoch != (args.max_epochs-1) and epoch >= semi_epoch and (epoch+1)%2==0 and epoch <=15:

                pseudo_records = _build_pseudo_from_grid(
                    model=model,
                    train_loader = get_split_loader(train_split,
                                                    training=True,
                                                    testing=args.testing,
                                                    weighted=args.weighted_sample,
                                                    mode=args.mode,
                                                    batch_size=args.batch_size),
                    remove_loader=remove_loader,
                    bins=bins_train,
                    args=args,
                    k=getattr(args, 'pseudo_topk', 6),
                    agree_thr=getattr(args, 'pseudo_agree_thr', 0.5),
                    hard_thr=getattr(args, 'pseudo_conf', 0.6),
                    device='cuda',
                    epoch=epoch,
                    ema_bank=ema_bank,   

                )


                if len(pseudo_records) > 0:
                    pseudo_ds = PseudoBagDataset(pseudo_records)
                    concat_ds = ConcatDataset([base_train_ds, pseudo_ds])
                    train_loader = DataLoader(
                        concat_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=collate_MIL_survival,
                        num_workers=0, pin_memory=True
                    )
                else:
                    pseudo_ds = None
                    train_loader = get_split_loader(train_split,
                                                    training=True,
                                                    testing=args.testing,
                                                    weighted=args.weighted_sample,
                                                    mode=args.mode,
                                                    batch_size=args.batch_size)  


    index = train_loss_list.index(min(train_loss_list))
    c_index_small_loss = c_index_list[index]
    c_index_final = c_index_list[-1]

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    # results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)

    print('Val c_index_small_loss: {:.4f} c_index_final {:.4f}'.format(c_index_small_loss, c_index_final))
    return c_index_small_loss, c_index_final


def train_loop_survival(args, epoch, model, loader, optimizer, 
                        scheduler, loss_fn=None, reg_fn=None, 
                        lambda_reg=0., gc=16, ema_bank=None, 
                        bank_momentum=0.9,gt_path_set=None):
    model.train()
    train_loss_surv, train_loss = 0., 0.
    print('\n')
    print('lr:',optimizer.param_groups[0]['lr'])


    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    hy = 0
    up_data = 0
    for batch_idx, (data_WSI, label, event_time, c, path) in enumerate(loader):
        if torch.equal(data_WSI, torch.ones(1)):
            hy += 1
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()
        c = c.cuda()

        if ema_bank is not None:
            hazards, S, div_loss, x_feat, attn_1 = model(x_path=data_WSI, return_bank=True)
        else:
            hazards, S, div_loss = model(x_path=data_WSI)


        loss =  loss_fn(hazards=hazards, S=S, Y=label, c=c) +  div_loss
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        loss = loss / gc + loss_reg
        loss.backward()
    # ⑤ EMA bank 

        key = path[0]  # batch_size=1
        if gt_path_set is not None and epoch == 0:
            gt_path_set.add(key)

        allow_update = (gt_path_set is None) or (key in gt_path_set)

        if ema_bank is not None and allow_update:
            up_data += 1
            ema_bank.update(
                paths=path,
                feats=x_feat.detach(),
                attns=attn_1.detach(),
                hazards=hazards.detach(),
                times=event_time,
                cens=c.detach(),
                momentum=bank_momentum
            )

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    scheduler.step()

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch,
                                                                                                 train_loss_surv,
                                                                                                 train_loss,
                                                                                                 c_index))
    return train_loss_surv


def validate_survival(epoch, model, loader, loss_fn=None, reg_fn=None, lambda_reg=0.):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, event_time, c, path) in enumerate(loader):
        if torch.equal(data_WSI, torch.ones(1)):
            continue
        if data_WSI.size(0) < 1000:
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()
        c = c.cuda()

        with torch.no_grad():
            hazards, S, _,_,_,_= model(x_path=data_WSI)  # return hazards, S, Y_hat, A_raw, results_dict
        
    #    print(f'{path}:{hazards}')
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]

    print('val/loss_surv, {}, {}'.format(val_loss_surv, epoch))
    print('val/c-index: {}, {}'.format(c_index, epoch))

    return c_index


def summary_survival(model, loader, n_classes):
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, label, event_time, c, _) in enumerate(loader):

        if torch.equal(data_WSI, torch.ones(1)):
            continue
        if data_WSI.size(0) < 1000:
            continue

        data_WSI = data_WSI.cuda()
        label = label.cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(x_path=data_WSI)

        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.asscalar(event_time)
        c = np.asscalar(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                         all_event_times,
                                         all_risk_scores,
                                         tied_tol=1e-08)[0]
    return patient_results, c_index


from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os
import torch

class PseudoBagDataset(Dataset):

    def __init__(self, records):
        """
        records: List[dict]:
        {
          'path': '/.../pt_files/<slide>.pt',
          'label': int(bin_idx),              
          'event_time': float(pseudo_months), 
          'c': int(0/1)                      
        }
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        path = r['path']
        if (not os.path.exists(path)):
            data_WSI = torch.ones(1)
        else:
            bag = torch.load(path)
            if not isinstance(bag, torch.Tensor):
                bag = torch.tensor(bag)
            data_WSI = bag
        return (data_WSI, int(r['label']), float(r['event_time']), int(r['c']), path)
