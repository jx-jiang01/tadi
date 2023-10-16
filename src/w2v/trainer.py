# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model2 import Trainer
import data
import dataset
import sklearn.metrics as metrics
import evaluate
from tqdm import tqdm
from nce_loss import NCELoss
import random
from datetime import datetime

# seed_list = [21, 3407, 42, 30, 35]
# random.seed(21)
seed = 21

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = "/root/autodl-tmp/"
data_root = root + "data/news/mind-small/"
model_root = "/root/autodl-nas/model/news/mind-small/v1/w2v/"
word_emb_path = data_root + "vocab_embs.txt"
news_paths = [data_root + "train/news.tsv", data_root + "dev/news.tsv"]
en_paths = [
    data_root + "train/entity_embedding.vec",
    data_root + "dev/entity_embedding.vec",
]
train_beh_path = data_root + "train/behaviors.tsv"
test_beh_path = data_root + "dev/behaviors.tsv"

cate_path = data_root + "cate_embs.vec"
subcate_path = data_root + "subcate_embs.vec"

# word_idx_path = root + "nlp/glove/word_idx.txt"
K = 4
click_num = 40
ti_max_len = 30
en_max_len = 10


def preprocess():
    news_paths = [data_root + "train/news.tsv"]
    glove_emb_path = root + "nlp/glove/glove.840B.300d/glove.840B.300d.txt"
    vocab_embs, vocab_idx = data.load_vocab_embs(glove_emb_path)
    data.preprocess_cate_subcate(vocab_embs, vocab_idx, cate_path, subcate_path)
    filtered_w2v, word_idx = data.filter_vocab_embedding(vocab_embs, vocab_idx, news_paths)
    data.save_w2v(word_emb_path, filtered_w2v, word_idx)


def prepare_data(news_paths, en_paths, train_beh_path, test_beh_path, cate_id_idx, 
    subcate_id_idx, word_idx=None):
    # 用户可能会不知道，但是新闻肯定知道
    en_id_idx, en_embs = data.load_entity_embedding(en_paths)

    news_id_idx, news_ti, news_cate, news_subcate, news_ti_en = data.load_mind_news_data(news_paths, 
        cate_id_idx, subcate_id_idx, en_id_idx, ti_max_len, en_max_len, word_idx)
    
    
    user_id_idx, \
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, labels = data.processed_data(train_beh_path, K, click_num, \
                ti_max_len, en_max_len, news_id_idx, news_ti, \
                    news_cate, news_subcate, news_ti_en)
    
    user_size = len(user_id_idx)
    news_size = len(news_id_idx)
    cate_size = len(cate_id_idx)
    sub_cate_size = len(subcate_id_idx)

    dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_subcate, \
        dev_clk_ti_en, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
            = data.processed_data(test_beh_path, K, 
                click_num, ti_max_len, en_max_len, news_id_idx, news_ti, \
                    news_cate, news_subcate, news_ti_en, user_id_idx, "dev", num=-1)

    return user_size, news_size, cate_size, sub_cate_size, en_embs, \
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, labels, \
                dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
                    dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors


def get_attr(model):
    cate_ti_attr = model.cate_ti_attr
    pass


def scoring(uemb, iemb):
    scores = np.matmul(uemb, np.transpose(iemb))
    scores = np.max(scores, axis=-1, keepdims=True)
    pred = np.sum(scores)
    return pred


lr = 4e-5
a = 0.8
b = 0.1
c = 0.1


def train():
    random.seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

    print("-" * 10, str(lr), "-" * 10)
    print(seed, click_num, ti_max_len, en_max_len)
    print(a, b, c)

    vocab_embs, vocab_idx = data.load_vocab_embs(word_emb_path, True)
    cate_id_idx, cate_embs = data.load_cate_embedding(cate_path)
    subcate_id_idx, subcate_embs = data.load_cate_embedding(subcate_path)
    
    print("w2v loaded")

    user_size, news_size, cate_size, sub_cate_size, en_embs, \
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, labels, \
                dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
                    dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
                        = prepare_data(news_paths, en_paths, train_beh_path, test_beh_path, cate_id_idx, 
                            subcate_id_idx, vocab_idx)

    batch_size = 32

    train_dataset = dataset.MindTrainDataset(user_idx, clk_news_idx, clk_ti, clk_cate, 
        clk_subcate, clk_ti_en, news_idx, ti, cate, subcate, ti_en)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_user_dataset = dataset.MindUserDataset(dev_sess_idx, dev_clk_ti, 
        dev_clk_cate, dev_clk_subcate, dev_clk_ti_en)
    dev_user_data_loader = DataLoader(dev_user_dataset, batch_size=batch_size)
    
    dev_news_dataset = dataset.MindNewsDataset(dev_news_idx, dev_ti, dev_cate, 
        dev_subcate, dev_ti_en)
    dev_news_data_loader = DataLoader(dev_news_dataset, batch_size=batch_size)
    print("data prepared")

    criterion = NCELoss()
    weight = torch.tensor([4])
    criterion1 = nn.BCELoss(weight=weight).to(device)

    vocab_embs = torch.tensor(np.array(vocab_embs), dtype=torch.float32)
    cate_embs = torch.tensor(np.array(cate_embs), dtype=torch.float32)
    subcate_embs = torch.tensor(np.array(subcate_embs), dtype=torch.float32)
    en_embs = torch.tensor(np.array(en_embs), dtype=torch.float32)

    model = Trainer(vocab_embs, cate_embs, subcate_embs, en_embs, click_num, 
        ti_max_len, en_max_len)
    model.to(device)

    epochs = 5
    iters = len(train_data_loader)
    # total_steps = iters * epochs
    quater = iters // 2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # pcg_optimizer = PCGrad(optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iters // 4, T_mult=2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=iters, epochs=epochs)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps)

    model_file = None
    max_auc = -1
    for epoch in range(epochs):
        _batch = 0
        for _, _, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en, _news_idx, _ti, \
            _cate, _subcate, _ti_en in train_data_loader:
            model.train()
            
            _clk_ti = _clk_ti.to(device)
            _clk_cate = _clk_cate.to(device)
            _clk_subcate = _clk_subcate.to(device)
            _clk_ti_en = _clk_ti_en.to(device)

            _ti = _ti.to(device)
            _cate = _cate.to(device)
            _subcate = _subcate.to(device)
            _ti_en = _ti_en.to(device)
            
#             start_time = datetime.now()
            pred, pred1, pred2 = model(_clk_ti, _clk_cate, _clk_subcate, 
                _clk_ti_en, _ti, _cate, _subcate, _ti_en)
#             end_time = datetime.now()
#             print(end_time - start_time)
#             return
            labels = torch.tensor([[[0], [0], [0], [0], [1]]] * pred.shape[0], dtype=torch.float32).to(device)
            loss0 = criterion(pred)
            loss1 = criterion1(pred1, labels)
            loss2 = criterion1(pred2, labels)

            loss = (a * loss0 + b * loss1 + c * loss2) / 3  #    (a * loss0 + b * loss1 + c * loss2) / 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # loss_list = [loss0, loss1, loss2]
            # pcgrad_fn(model, loss_list, optimizer)
            # optimizer.step()
            # break
            _batch += 1
           
            if _batch % quater == 0:
                model.eval()
                with torch.no_grad():
                    # news encoding
                    user_embeddings, news_embeddings = {}, {}
                    for _news_idx, _ti, _cate, _subcate, _ti_en in dev_news_data_loader:
                        
                        _ti = _ti.to(device)
                        _cate = _cate.to(device)
                        _subcate = _subcate.to(device)
                        _ti_en = _ti_en.to(device)

                        iemb, _, _, _ = model.news_embedding(_ti, _cate, _subcate, _ti_en)

                        _news_idx = _news_idx.detach().cpu().numpy().tolist()

                        for i, _idx in enumerate(_news_idx):
                            news_embeddings[_idx] = iemb[i].reshape(-1).detach().cpu().numpy()
                            # np.array([
                            #     iemb[i].reshape(-1).detach().cpu().numpy(),
                            #     icate_part[i].reshape(-1).detach().cpu().numpy(),
                            #     isubcate_part[i].reshape(-1).detach().cpu().numpy()
                            # ])
                    
                    # user encoding
                    for _sess_idx, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en in dev_user_data_loader:
                        _clk_ti = _clk_ti.to(device)
                        _clk_cate = _clk_cate.to(device)
                        _clk_subcate = _clk_subcate.to(device)
                        _clk_ti_en = _clk_ti_en.to(device)

                        uemb, _, _, _ = model.user_embedding(_clk_ti, _clk_cate, _clk_subcate, _clk_ti_en)
                        
                        _sess_idx = _sess_idx.detach().cpu().numpy().tolist()
                        for i, _idx in enumerate(_sess_idx):
                            user_embeddings[_idx] = uemb[i].reshape(-1).detach().cpu().numpy()
                            # np.array([
                            #     uemb[i].reshape(-1).detach().cpu().numpy(),
                            #     ucate_part[i].reshape(-1).detach().cpu().numpy(),
                            #     usubcate_part[i].reshape(-1).detach().cpu().numpy()
                            # ])
                        
                auc, mrr, ndcg5, ndcg10 = [], [], [], []
                for _sess_idx, _news_idx, y_true in dev_sess_behaviors:
                    uemb = user_embeddings[_sess_idx]
                    y_score = []
                    for _nidx in _news_idx:
                        nemb = news_embeddings[_nidx] 
                        res = np.dot(uemb, nemb)
                        y_score.append(res)
                    _auc = metrics.roc_auc_score(y_true, y_score)
                    _mrr = evaluate.mrr_score(y_true, y_score)
                    _ndcg5 = evaluate.ndcg_score(y_true, y_score, k=5)
                    _ndcg10 = evaluate.ndcg_score(y_true, y_score, k=10)
                    auc.append(_auc)
                    mrr.append(_mrr)
                    ndcg5.append(_ndcg5)
                    ndcg10.append(_ndcg10)
                auc = np.array(auc).mean()
                mrr = np.array(mrr).mean()
                ndcg5 = np.array(ndcg5).mean()
                ndcg10 = np.array(ndcg10).mean()
                print(epoch, auc, mrr, ndcg5, ndcg10)
                if auc > max_auc:
                    if model_file is not None and os.path.exists(model_file):
                        os.remove(model_file)
                    max_auc = round(auc, 4)
                    model_file = model_root + "w2v_epoch={}_lr={}_batch={}_auc={}.pt".format(str(epoch), str(lr), str(_batch), 
                        str(max_auc))
                    torch.save(model, model_file)


if __name__ == "__main__":
    # model = torch.load(model_root + "epoch=4_batch=7386_auc=0.6836.pt")
    # print(model)
    # data.split_w2v(word_embedding_path, word_idx_path)
    # preprocess()
#     a_b_c_list = [
#         [0.8, 0.01, 0.01],
#         [0.8, 0.1, 0.1],
#         [0.8, 0.05, 0.05],
        
#         [0.8, 0.005, 0.005],
#     ]
    seeds = [76, 56]  # 76 21 
    for _seed in seeds:
        seed = _seed
        train()
