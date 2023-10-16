# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model3 import Trainer
import data_bert
import dataset
import sklearn.metrics as metrics
import evaluate
from tqdm import tqdm
from nce_loss import NCELoss
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from scipy.stats import rankdata
import random

seed = 51
random.seed(seed)
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = "/root/autodl-tmp/"
data_root = root + "data/news/mind-large/"
model_root = "/root/autodl-nas/model/news/mind-large/bert/"
word_emb_path = data_root + "vocab_embs.txt"
bert_path = root + "nlp/all-MiniLM-L12-v2/"
tokenizer = BertTokenizer.from_pretrained(bert_path)

news_paths = [data_root + "train/news.tsv", data_root + "dev/news.tsv", data_root + "test/news.tsv"]
en_paths = [
    data_root + "train/entity_embedding.vec",
    data_root + "dev/entity_embedding.vec",
    data_root + "test/entity_embedding.vec",
]

train_beh_path = data_root + "train/behaviors.tsv"
dev_beh_path = data_root + "dev/behaviors.tsv"
test_beh_path = data_root + "test/behaviors.tsv"

cate_path = data_root + "cate_embs.vec"
subcate_path = data_root + "subcate_embs.vec"

news_ti_embs_dict = {}

# word_idx_path = root + "nlp/glove/word_idx.txt"
K = 4
click_num = 40
ti_max_len = 32
en_max_len = 10


def preprocess():
    glove_emb_path = root + "nlp/glove/glove.840B.300d/glove.840B.300d.txt"
    data_bert.preprocess_cate_subcate(glove_emb_path, cate_path, subcate_path)
    # filtered_w2v, word_idx = data_bert.filter_vocab_embedding(glove_emb_path, news_paths)
    # data_bert.save_w2v(word_emb_path, filtered_w2v, word_idx)


def prepare_data(news_paths, en_paths, train_beh_path, test_beh_path, cate_id_idx, 
    subcate_id_idx, tokenizer):
    # 用户可能会不知道，但是新闻肯定知道
    en_id_idx, en_embs = data_bert.load_entity_embedding(en_paths)

    news_id_idx, news_ti, news_cate, news_subcate, news_ti_en = data_bert.load_mind_news_data(news_paths, 
        cate_id_idx, subcate_id_idx, en_id_idx, ti_max_len, en_max_len)
    
    for key in news_ti:
        news_ti_embs_dict[news_id_idx[key]] = tokenizer(news_ti[key],
                max_length=ti_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

    user_id_idx, \
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, labels = data_bert.processed_data(train_beh_path, K, click_num, \
                ti_max_len, en_max_len, news_id_idx, news_ti, \
                    news_cate, news_subcate, news_ti_en)
    
    user_size = len(user_id_idx)
    news_size = len(news_id_idx)
    cate_size = len(cate_id_idx)
    sub_cate_size = len(subcate_id_idx)

    dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_subcate, \
        dev_clk_ti_en, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
            = data_bert.processed_data(test_beh_path, K, 
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


def train_custom_collate(batch):
    user_idx, clk_news_idx, clk_ti, clk_cate, \
    clk_subcate, clk_ti_en, news_idx, ti, cate, subcate, ti_en = [], [], [], [], [], [], [], [], [], [], []

    clk_ti_input_ids, clk_ti_token_type_ids, clk_ti_attention_mask = [], [], []
    ti_input_ids, ti_token_type_ids, ti_attention_mask = [], [], []

    for _batch in batch:
        user_idx.append(_batch[0])
        clk_news_idx.append(_batch[1])

        for _nid in _batch[1]:
            _embs = news_ti_embs_dict[_nid]
            clk_ti_input_ids.append(_embs["input_ids"])
            clk_ti_token_type_ids.append(_embs["token_type_ids"])
            clk_ti_attention_mask.append(_embs["attention_mask"])

        clk_cate.append(_batch[2])
        clk_subcate.append(_batch[3])
        clk_ti_en.append(_batch[4])
        news_idx.append(_batch[5])

        for _nid in _batch[5]:
            _embs = news_ti_embs_dict[_nid]
            ti_input_ids.append(_embs["input_ids"])
            ti_token_type_ids.append(_embs["token_type_ids"])
            ti_attention_mask.append(_embs["attention_mask"])

        cate.append(_batch[6])
        subcate.append(_batch[7])
        ti_en.append(_batch[8])
    
    clk_ti_data = {
        "input_ids": torch.stack(clk_ti_input_ids).squeeze(-2),
        "token_type_ids": torch.stack(clk_ti_token_type_ids).squeeze(-2),
        "attention_mask": torch.stack(clk_ti_attention_mask).squeeze(-2)
    }
    clk_ti = BatchEncoding(clk_ti_data)

    ti_data = {
        "input_ids": torch.stack(ti_input_ids).squeeze(-2),
        "token_type_ids": torch.stack(ti_token_type_ids).squeeze(-2),
        "attention_mask": torch.stack(ti_attention_mask).squeeze(-2)
    }
    ti = BatchEncoding(ti_data)
    return user_idx, clk_news_idx, clk_ti, torch.stack(clk_cate), \
        torch.stack(clk_subcate), torch.stack(clk_ti_en), news_idx, ti, \
            torch.stack(cate), torch.stack(subcate), torch.stack(ti_en)


def news_custom_collate(batch):
    news_idx, ti, cate, subcate, ti_en = [], [], [], [], []
    ti_input_ids, ti_token_type_ids, ti_attention_mask = [], [], []
    for _batch in batch:
        news_idx.append(_batch[0])
        cate.append(_batch[1])
        subcate.append(_batch[2])
        ti_en.append(_batch[3])

        _embs = news_ti_embs_dict[_batch[0]]
        ti_input_ids.append(_embs["input_ids"])
        ti_token_type_ids.append(_embs["token_type_ids"])
        ti_attention_mask.append(_embs["attention_mask"])
    
    ti_data = {
        "input_ids": torch.stack(ti_input_ids).squeeze(-2),
        "token_type_ids": torch.stack(ti_token_type_ids).squeeze(-2),
        "attention_mask": torch.stack(ti_attention_mask).squeeze(-2)
    }
    ti = BatchEncoding(ti_data)
    return news_idx, ti, torch.stack(cate), torch.stack(subcate), torch.stack(ti_en)


def user_custom_collate(batch):
    sess_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en = [], [], [], [], []
    clk_ti_input_ids, clk_ti_token_type_ids, clk_ti_attention_mask = [], [], []
    for _batch in batch:
        sess_idx.append(_batch[0])

        for _nid in _batch[1]:
            _embs = news_ti_embs_dict[_nid]
            clk_ti_input_ids.append(_embs["input_ids"])
            clk_ti_token_type_ids.append(_embs["token_type_ids"])
            clk_ti_attention_mask.append(_embs["attention_mask"])

        clk_cate.append(_batch[2])
        clk_subcate.append(_batch[3])
        clk_ti_en.append(_batch[4])
    
    clk_ti_data = {
        "input_ids": torch.stack(clk_ti_input_ids).squeeze(-2),
        "token_type_ids": torch.stack(clk_ti_token_type_ids).squeeze(-2),
        "attention_mask": torch.stack(clk_ti_attention_mask).squeeze(-2)
    }
    clk_ti = BatchEncoding(clk_ti_data)

    return sess_idx, clk_ti, torch.stack(clk_cate), torch.stack(clk_subcate), \
        torch.stack(clk_ti_en)


def train():
    cate_id_idx, cate_embs = data_bert.load_cate_embedding(cate_path)
    subcate_id_idx, subcate_embs = data_bert.load_cate_embedding(subcate_path)
    
    print("w2v loaded")

    user_size, news_size, cate_size, sub_cate_size, en_embs, \
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, labels, \
                dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
                    dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
                        = prepare_data(news_paths, en_paths, train_beh_path, dev_beh_path, cate_id_idx, 
                            subcate_id_idx, tokenizer)

    print("data prepared")
    lr, max_lr = 1e-7, 5e-6
    print("-" * 10, str(lr), "-" * 10)
    
    criterion = NCELoss()
    weight = torch.as_tensor(np.array([4]))
    criterion1 = nn.BCELoss(weight=weight).to(device)

    cate_embs = torch.as_tensor(np.array(cate_embs), dtype=torch.float32)
    subcate_embs = torch.as_tensor(np.array(subcate_embs), dtype=torch.float32)
    en_embs = torch.as_tensor(np.array(en_embs), dtype=torch.float32)
    
    batch_size = 16

    train_dataset = dataset.MindBertTrainDataset(user_idx, clk_news_idx, clk_cate, 
        clk_subcate, clk_ti_en, news_idx, cate, subcate, ti_en)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=train_custom_collate)

    dev_user_dataset = dataset.MindBertUserDataset(dev_sess_idx, dev_clk_news_idx, 
        dev_clk_cate, dev_clk_subcate, dev_clk_ti_en)
    dev_user_data_loader = DataLoader(dev_user_dataset, batch_size=batch_size, 
        collate_fn=user_custom_collate)
    
    dev_news_dataset = dataset.MindBertNewsDataset(dev_news_idx, dev_cate, 
        dev_subcate, dev_ti_en)
    dev_news_data_loader = DataLoader(dev_news_dataset, batch_size=batch_size, 
        collate_fn=news_custom_collate)

    model = Trainer(bert_path, cate_embs, subcate_embs, en_embs, click_num, 
        ti_max_len, en_max_len)
    model.to(device)
    # freeze_layers = {"layer.0", "layer.1", "layer.2"}
    # for name, param in model.named_parameters():
    #     for key in freeze_layers:
    #         if key in name:
    #             param.requires_grad = False

    epochs = 2
    batch_num = len(train_data_loader)
    quater = batch_num // 2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=batch_num, 
        epochs=epochs)

    model_file = None
    max_auc = -1
    for epoch in range(epochs):
        _batch = 0
        for _, _, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en, _news_idx, _ti, \
        _cate, _subcate, _ti_en in train_data_loader:
            model.train()
            optimizer.zero_grad()
            _clk_ti = _clk_ti.to(device)
            _clk_cate = _clk_cate.to(device)
            _clk_subcate = _clk_subcate.to(device)
            _clk_ti_en = _clk_ti_en.to(device)

            _ti = _ti.to(device)
            _cate = _cate.to(device)
            _subcate = _subcate.to(device)
            _ti_en = _ti_en.to(device)

            pred, pred1, pred2 = model(_clk_ti, _clk_cate, _clk_subcate, 
                _clk_ti_en, _ti, _cate, _subcate, _ti_en)
            labels = torch.as_tensor(np.array([[[0], [0], [0], [0], [1]]] * pred.shape[0]), dtype=torch.float32).to(device)
            loss0 = criterion(pred)
            loss1 = criterion1(pred1, labels)
            loss2 = criterion1(pred2, labels)

            a = 0.8
            b = 0.1
            c = 0.1

            loss = (a * loss0 + b * loss1) / 3  #  + c * loss2
            # print(loss0, loss1, loss2)
            loss.backward()
            optimizer.step()
            # break
            _batch += 1
            scheduler.step()

            if _batch % quater == 0:
                model.eval()

                user_embeddings, news_embeddings = {}, {}

                with torch.no_grad():
                    # news encoding
                    for _news_idx, _ti, _cate, _subcate, _ti_en in dev_news_data_loader:
                        
                        _ti = _ti.to(device)
                        _cate = _cate.to(device)
                        _subcate = _subcate.to(device)
                        _ti_en = _ti_en.to(device)

                        iemb, _, _, _ = model.news_embedding(_ti, _cate, _subcate, _ti_en)

                        for i, _idx in enumerate(_news_idx):
                            news_embeddings[_idx] = iemb[i].reshape(-1).detach().cpu().numpy()
                        
                    # user encoding
                    for _sess_idx, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en in dev_user_data_loader:
                        _clk_ti = _clk_ti.to(device)
                        _clk_cate = _clk_cate.to(device)
                        _clk_subcate = _clk_subcate.to(device)
                        _clk_ti_en = _clk_ti_en.to(device)

                        uemb, _, _, _ = model.user_embedding(_clk_ti, _clk_cate, _clk_subcate, _clk_ti_en)
                        
                        for i, _idx in enumerate(_sess_idx):
                            user_embeddings[_idx] = uemb[i].reshape(-1).detach().cpu().numpy()
                
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
                    model_file = model_root + "bert_epoch={}_batch={}_auc={}.pt".format(str(epoch), str(_batch), 
                        str(max_auc))
                    torch.save(model, model_file)


def test_prepare_data(news_paths, en_paths, test_beh_path, cate_id_idx, 
    subcate_id_idx, tokenizer):
    # 用户可能会不知道，但是新闻肯定知道
    en_id_idx, en_embs = data_bert.load_entity_embedding(en_paths)

    news_id_idx, news_ti, news_cate, news_subcate, news_ti_en = data_bert.load_mind_news_data(news_paths, 
        cate_id_idx, subcate_id_idx, en_id_idx, ti_max_len, en_max_len)
    
    for key in news_ti:
        news_ti_embs_dict[news_id_idx[key]] = tokenizer(news_ti[key],
                max_length=ti_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
    
    news_size = len(news_id_idx)
    cate_size = len(cate_id_idx)
    sub_cate_size = len(subcate_id_idx)

    dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_subcate, \
        dev_clk_ti_en, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
            = data_bert.processed_data(test_beh_path, K, 
                click_num, ti_max_len, en_max_len, news_id_idx, news_ti, \
                    news_cate, news_subcate, news_ti_en, data_type="test")

    return news_size, cate_size, sub_cate_size, en_embs, \
            dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
                dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors



def test():
    model = torch.load(model_root + "bert_epoch=1_batch=211478_auc=0.7008.pt")
    model.to(device)
    cate_id_idx, _ = data_bert.load_cate_embedding(cate_path)
    subcate_id_idx, _ = data_bert.load_cate_embedding(subcate_path)
    
    print("w2v loaded")

    news_size, cate_size, sub_cate_size, en_embs, \
        dev_sess_idx, dev_user_idx, dev_clk_news_idx, dev_clk_ti, dev_clk_cate, dev_clk_ti_en, \
            dev_clk_subcate, dev_news_idx, dev_ti, dev_cate, dev_subcate, dev_ti_en, dev_sess_behaviors \
                = test_prepare_data(news_paths, en_paths, test_beh_path, cate_id_idx, 
                    subcate_id_idx, tokenizer)

    batch_size = 16

    dev_user_dataset = dataset.MindBertUserDataset(dev_sess_idx, dev_clk_news_idx, 
        dev_clk_cate, dev_clk_subcate, dev_clk_ti_en)
    dev_user_data_loader = DataLoader(dev_user_dataset, batch_size=batch_size, 
        collate_fn=user_custom_collate, num_workers=4)
    
    dev_news_dataset = dataset.MindBertNewsDataset(dev_news_idx, dev_cate, 
        dev_subcate, dev_ti_en)
    dev_news_data_loader = DataLoader(dev_news_dataset, batch_size=batch_size, 
        collate_fn=news_custom_collate, num_workers=4)
    
    model.eval()

    user_embeddings, news_embeddings = {}, {}

    with torch.no_grad():
        # news encoding
        for _news_idx, _ti, _cate, _subcate, _ti_en in dev_news_data_loader:
            
            _ti = _ti.to(device)
            _cate = _cate.to(device)
            _subcate = _subcate.to(device)
            _ti_en = _ti_en.to(device)

            iemb, _, _, _ = model.news_embedding(_ti, _cate, _subcate, _ti_en)

            for i, _idx in enumerate(_news_idx):
                news_embeddings[_idx] = iemb[i].reshape(-1).detach().cpu().numpy()
            
        # user encoding
        for _sess_idx, _clk_ti, _clk_cate, _clk_subcate, _clk_ti_en in dev_user_data_loader:
            _clk_ti = _clk_ti.to(device)
            _clk_cate = _clk_cate.to(device)
            _clk_subcate = _clk_subcate.to(device)
            _clk_ti_en = _clk_ti_en.to(device)

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
        
    
    with open("/root/autodl-nas/model/news/mind-large/test/prediction.txt", "w") as f:
        for pred in predictions:
            f.write(" ".join(pred) + "\n")
        f.close()
    print("done")


if __name__ == "__main__":
    # data.split_w2v(word_embedding_path, word_idx_path)
    # preprocess()
    # ti = ["[PAD]", "[PAD]"]
    # ti_embs = tokenizer(ti,
    #                 max_length=32,
    #                 padding="max_length",
    #                 truncation=True,
    #                 return_tensors="pt")
    # ti_list = []
    # for _ti in ti:
    #     ti_list.append(tokenizer(_ti,
    #                 max_length=32,
    #                 padding="max_length",
    #                 truncation=True,
    #                 return_tensors="pt")
    #     )
    # input_ids, token_type_ids, attention_mask = [], [], []
    # for _ti in ti_list:
    #     input_ids.append(_ti["input_ids"])
    #     token_type_ids.append(_ti["token_type_ids"])
    #     attention_mask.append(_ti["attention_mask"])

    # clk_ti_data = {
    #     "input_ids": torch.stack(input_ids).squeeze(-2),
    #     "token_type_ids": torch.stack(token_type_ids).squeeze(-2),
    #     "attention_mask": torch.stack(attention_mask).squeeze(-2)
    # }
    
    # encoding = BatchEncoding(clk_ti_data)
    # print()
#     train()
    test()
