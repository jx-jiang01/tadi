# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model2 import Trainer
import random
import data
import dataset
import sklearn.metrics as metrics
import evaluate
from tqdm import tqdm
from nce_loss import NCELoss
from transformers import MobileBertTokenizer, MobileBertModel, BertTokenizer, BertModel
from scipy.stats import rankdata


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = "/root/autodl-tmp/"
data_root = root + "data/news/mind-large/"
model_root = "/root/autodl-nas/model/news/mind-large/"

news_paths = [data_root + "train/news.tsv", data_root + "dev/news.tsv", data_root + "test/news.tsv"]
en_paths = [
    data_root + "train/entity_embedding.vec",
    data_root + "dev/entity_embedding.vec",
    data_root + "test/entity_embedding.vec"
]
test_beh_path = data_root + "test/behaviors.tsv"

word_emb_path = data_root + "vocab_embs.txt"
cate_path = data_root + "cate_embs.vec"
subcate_path = data_root + "subcate_embs.vec"


K = 4
click_num = 40
ti_max_len = 30
en_max_len = 10


def prepare_data(news_paths, en_paths, beh_path, cate_id_idx, 
    subcate_id_idx, word_idx=None):
    # 用户可能会不知道，但是新闻肯定知道

    en_id_idx, en_embs = data.load_entity_embedding(en_paths)

    news_id_idx, news_ti, news_cate, news_subcate, news_ti_en = data.load_mind_news_data(news_paths, 
        cate_id_idx, subcate_id_idx, en_id_idx, ti_max_len, en_max_len, word_idx)
    

    dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_subcate, \
        dev_clk_ti_en, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
            = data.processed_data(beh_path, K, 
                click_num, ti_max_len, en_max_len, news_id_idx, news_ti, \
                    news_cate, news_subcate, news_ti_en, data_type="test")

    return dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
                dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors


def get_attr(model):
    cate_ti_attr = model.cate_ti_attr
    pass


def test():
    

    vocab_embs, vocab_idx = data.load_vocab_embs(word_emb_path, True)
#     if "It" not in vocab_idx:
#         print("It")
#     if "it" not in vocab_idx:
#         print("it")
    cate_id_idx, cate_embs = data.load_cate_embedding(cate_path)
    subcate_id_idx, subcate_embs = data.load_cate_embedding(subcate_path)

    print("w2v loaded")

    test_news_paths = [data_root + "train/news.tsv", data_root + "dev/news.tsv", data_root + "test/news.tsv"]

    dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
        dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
            = prepare_data(test_news_paths, en_paths, test_beh_path, cate_id_idx, 
                subcate_id_idx, vocab_idx)

#     idx_vocab = {}
#     for vocab in vocab_idx:
#         idx = vocab_idx[vocab]
#         idx_vocab[idx] = vocab
#     cate_idx_id, subcate_idx_id = {}, {}
#     for cate_id in cate_id_idx:
#         idx = cate_id_idx[cate_id]
#         cate_idx_id[idx] = cate_id
#     for subcate_id in subcate_id_idx:
#         idx = subcate_id_idx[subcate_id]
#         subcate_idx_id[idx] = subcate_id
#     for _cate, _subcate, _ti in zip(dev_cate[: 10], dev_subcate[: 10], dev_ti[: 10]):
#         print("\"" + cate_idx_id[_cate] + "\",")
#         print("\"" + subcate_idx_id[_subcate] + "\",")
#         print("\"" + "||".join([idx_vocab[v] for v in _ti]) + "\"")

    dev_user_dataset = dataset.MindUserDataset(dev_sess_idx, dev_clk_ti, 
        dev_clk_cate, dev_clk_subcate, dev_clk_ti_en)
    dev_user_data_loader = DataLoader(dev_user_dataset, batch_size=128)
    
    dev_news_dataset = dataset.MindNewsDataset(dev_news_idx, dev_ti, dev_cate, 
        dev_subcate, dev_ti_en)
    dev_news_data_loader = DataLoader(dev_news_dataset, batch_size=128)
    print("data prepared")


    model = torch.load("/root/autodl-nas/model/news/mind-large/glove/" + "embed_type=w2v_task_type=recall_epoch=4_lr=1e-06_batch=105740_auc=0.7011.pt", map_location=torch.device(device))
    # model.to(device)
    model.eval()

    user_embeddings, news_embeddings = {}, {}
    with torch.no_grad():
        # news encoding
        for _news_idx, _ti, _cate, _subcate, _ti_en in dev_news_data_loader:
            
            _ti = _ti.to(device)
            _cate = _cate.to(device)
            _subcate = _subcate.to(device)
            _ti_en = _ti_en.to(device)

            _news_idx = _news_idx.detach().cpu().numpy().tolist()

            iemb, _, _, _ = model.news_embedding(_ti, _cate, _subcate, _ti_en)

            for i, _idx in enumerate(_news_idx):
                news_embeddings[_idx] = iemb[i].reshape(-1).detach().cpu().numpy()
                
        # user encoding
        for _sess_idx, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en in dev_user_data_loader:
            _clk_ti = _clk_ti.to(device)
            _clk_cate = _clk_cate.to(device)
            _clk_subcate = _clk_subcate.to(device)
            _clk_ti_en = _clk_ti_en.to(device)
            
            _sess_idx = _sess_idx.detach().cpu().numpy().tolist()

            uemb, _, _, _ = model.user_embedding(_clk_ti, _clk_cate, _clk_subcate, _clk_ti_en)
            
            for i, _idx in enumerate(_sess_idx):
                user_embeddings[_idx] = uemb[i].reshape(-1).detach().cpu().numpy()
            
    
    predictions = []
    for _sess_idx, _news_idx, _ in dev_sess_behaviors:
        uemb = user_embeddings[_sess_idx]
        y_score = []
        for _nidx in _news_idx:
            nemb = news_embeddings[_nidx] 
            res = np.dot(uemb, nemb)
            y_score.append(res)
        array = np.array(y_score)
        rank = - rankdata(array) + len(array) + 1
        rank = rank.tolist()
        predictions.append([str(_sess_idx), "[" + ",".join([str(int(v)) for v in rank]) + "]"])
        
    
    with open(model_root + "test/prediction76.txt", "w") as f:
        for pred in predictions:
            f.write(" ".join(pred) + "\n")
        f.close()
    print("done")


def transform():
    beh_path = data_root + "test/behaviors.tsv"
    session_ids, user_ids, clicks, impressions = data.load_mind_impression_data(beh_path)
    predictions = []
    with open(model_root + "test/prediction.csv") as f:
        line = f.readline()
        while line:
            terms = line.split(" ")
            predictions.append(terms[1])
            line = f.readline()
    with open(model_root + "test/prediction1.csv", "w") as f:
        for session_id, pred in zip(session_ids, predictions):
            f.write(" ".join([session_id, pred]) + "\n")
        f.close()


if __name__ == "__main__":
    # data.split_w2v(word_embedding_path, word_idx_path)
    # preprocess()
#     transform()
    # y_score = [4,1,3,2]
    # array = np.array(y_score)
    # rank = - rankdata(array) + len(array) + 1
    # print(rank.tolist())
    test()
