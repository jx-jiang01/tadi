# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
import os

dropout = 0.3
word_dim = 300
en_dim = 100
dim = 768
final_dim = 1024


class Trainer(nn.Module):

    def __init__(self, w2v, cate_embeddings, subcate_embeddings, en_embeddings, 
        click_num, ti_max_len, en_max_len) -> None:
        super(Trainer, self).__init__()
        self.user_encoder = UserEncoder(click_num, ti_max_len, en_max_len)
        self.news_encoder = NewsEncoder(ti_max_len, en_max_len)

        self.uinput_embedding = InputEmbedding(w2v, cate_embeddings, subcate_embeddings, en_embeddings, type="user")
        self.ninput_embedding = InputEmbedding(w2v, cate_embeddings, subcate_embeddings, en_embeddings, type="news")

        self.uemb_iemb_cat_cat_fc = nn.Sequential(
            nn.Linear((dim * 2 * 2 + en_dim + word_dim * 3) * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.uemb_iemb_cat_cat_fc2 = nn.Sequential(
            nn.Linear((dim * 2) * 2, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, dim)
        )

        self.uemb_iemb_cat_fc = nn.Sequential(
            nn.Linear(dim + dim * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, clk_ti, clk_cate, clk_subcate, clk_ti_en, ti, cate, subcate, ti_en):
        """双塔"""
        """用户测"""
        uemb, uemb_cat, ucate_part, usubcate_part = self.user_embedding(clk_ti, clk_cate, 
            clk_subcate, clk_ti_en)
        
        """物料侧"""
        iemb, iemb_cat, icate_part, isubcate_part = self.news_embedding(ti, cate, subcate, ti_en)
        
        """预测"""
        uemb = uemb.repeat([1, iemb.shape[1], 1])
        product = uemb * iemb
        pred = torch.sum(product, dim=-1, keepdim=True)

        """交互模块，为了用户和物料信息尽早交互"""
        
        # part 1
        uemb_cat = uemb_cat.repeat([1, iemb_cat.shape[1], 1])
        uemb_iemb_cat_cat = torch.cat([uemb_cat, iemb_cat], dim=-1)
        pred1 = self.uemb_iemb_cat_cat_fc(uemb_iemb_cat_cat)

        # part 2
        usubcate_part = usubcate_part.repeat([1, isubcate_part.shape[1], 1])
        ucate_part = ucate_part.repeat([1, icate_part.shape[1], 1])
        subcate_part = torch.cat([usubcate_part, isubcate_part], dim=-1)
        cate_part = torch.cat([ucate_part, icate_part], dim=-1)
        uemb_iemb_cat_cat2 = self.uemb_iemb_cat_cat_fc2(subcate_part)
        uemb_iemb_cat = torch.cat([cate_part, uemb_iemb_cat_cat2], dim=-1)
        pred2 = self.uemb_iemb_cat_fc(uemb_iemb_cat)
        
        return pred, pred1, pred2

    def user_embedding(self, clk_ti, clk_cate, clk_subcate, clk_ti_en):
        clk_ti_emb, clk_cate_emb, clk_subcate_emb, clk_ti_en_emb = self.uinput_embedding(clk_ti, clk_cate, 
            clk_subcate, clk_ti_en)
        return self.user_encoder(clk_ti_emb, clk_cate_emb, clk_subcate_emb, clk_ti_en_emb)
    
    def news_embedding(self, ti, cate, subcate, ti_en):
        ti_emb, cate_emb, subcate_emb, ti_en_emb = self.ninput_embedding(ti, cate, subcate, ti_en)
        return self.news_encoder(ti_emb, cate_emb, subcate_emb, ti_en_emb)


class InputEmbedding(nn.Module):
    def __init__(self, bert_path, cate_embeddings, subcate_embeddings, en_embeddings, type) -> None:
        super(InputEmbedding, self).__init__()
        self.word_embedding = BertModel.from_pretrained(bert_path)
        self.word_fc = nn.Sequential(
            nn.Linear(384 * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, word_dim)
        )

        # self.subcate_fc = nn.Sequential(
        #     nn.Linear(300, 512),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, word_dim)
        # )

        # self.ti_en_fc = nn.Sequential(
        #     nn.Linear(100, 512),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, word_dim)
        # )

        self.cate_embedding = nn.Embedding.from_pretrained(cate_embeddings, freeze=False)
        self.subcate_embedding = nn.Embedding.from_pretrained(subcate_embeddings, freeze=False)
        self.en_embedding = nn.Embedding.from_pretrained(en_embeddings, freeze=False)
        self.embed_dropout = nn.Dropout(p=dropout)

    def forward(self, ti, cate, subcate, ti_en):
        cate_emb = self.cate_embedding(cate)
        cate_emb = self.embed_dropout(cate_emb)
        subcate_emb = self.subcate_embedding(subcate)
        subcate_emb = self.embed_dropout(subcate_emb)
        ti_en_emb = self.en_embedding(ti_en)
        ti_en_emb = self.embed_dropout(ti_en_emb)

        hidden_states = self.word_embedding(**ti, output_hidden_states=True).hidden_states
        last_layer = hidden_states[-1]
        second_last_layer = hidden_states[-2]
        third_last_layer = hidden_states[-3]
        # first_layer = outputs[0]
        # second_layer = outputs[1]
        # third_layer = outputs[2]
        word_emb = torch.cat([
            last_layer, 
            second_last_layer, 
            third_last_layer,
            # last_layer[:, 1: -1, :], 
            # second_last_layer[:, 1: -1, :], 
            # third_last_layer[:, 1: -1, :],
            # first_layer[:, 1: -1, :],
            # second_layer[:, 1: -1, :],
            # third_layer[:, 1: -1, :]
        ], dim=-1)
        a, b = cate_emb.shape[0], cate_emb.shape[1]
        word_emb = self.word_fc(word_emb)
        c, d = word_emb.shape[1], word_emb.shape[2]
        ti_emb = self.embed_dropout(word_emb.reshape(a, b, c, d))
        
        return ti_emb, cate_emb, subcate_emb, ti_en_emb


class UserEncoder(nn.Module):
    def __init__(self, click_num, ti_max_len, en_max_len) -> None:
        super(UserEncoder, self).__init__()
        
        # self.embed_dropout = nn.Dropout(p=dropout)

        # self.cate_embedding = nn.Embedding.from_pretrained(cate_embeddings, freeze=False)
        # self.subcate_embedding = nn.Embedding.from_pretrained(subcate_embeddings, freeze=False)
        # self.word_embedding = nn.Embedding.from_pretrained(w2v, freeze=False)
        # self.en_embedding = nn.Embedding.from_pretrained(en_embeddings, freeze=False)
        # self.dropout = nn.Dropout(dropout)

        self.ti_bigru = Sequence(word_dim, word_dim, ti_max_len)
        # self.co_bigru = Sequence(self.word_dim, self.word_dim, co_max_len)
        self.ti_en_bigru = Sequence(en_dim, en_dim, en_max_len)
        # self.co_en_bigru = Sequence(self.en_dim, self.en_dim, en_max_len)

        self.ti_news_bigru = Sequence(word_dim, word_dim, click_num)
        self.ti_news_en_bigru = Sequence(en_dim, en_dim, click_num)

        self.cate_ti_attr = Attention(word_dim, word_dim, word_dim, hidden_dim=256, output_dim=dim)
        self.subcate_ti_attr = Attention(word_dim, word_dim, word_dim, hidden_dim=256, output_dim=dim)

        # self.cate_co_attr = Attention(self.word_dim, self.word_dim, self.word_dim, self.dim)
        # self.subcate_co_attr = Attention(self.word_dim, self.word_dim, self.word_dim, self.dim)

        # self.ti_en_self_attr = Attention(self.en_dim, self.en_dim, self.en_dim, self.dim)
        # self.co_en_self_attr = Attention(self.en_dim, self.en_dim, self.en_dim, self.dim)

        # self.cate_co_en_attr = Attention(self.word_dim, self.en_dim, self.en_dim, self.dim)
        # self.subcate_co_en_attr = Attention(self.word_dim, self.en_dim, self.en_dim, self.dim)

        self.clk_cate_bigru = Sequence(word_dim, word_dim, click_num)
        self.clk_subcate_bigru = Sequence(word_dim, word_dim, click_num)
        # self.clk_ti_en_bigru = Sequence(self.en_dim, self.en_dim, click_num)
        # self.clk_co_en_bigru = Sequence(self.en_dim, self.en_dim, click_num)

        # 
        # self.clk_cate_ti_gru = Sequence(self.dim, self.dim, click_num)
        # self.clk_subcate_ti_gru = Sequence(self.dim, self.dim, click_num)

        # self.clk_cate_co_gru = Sequence(self.dim, self.dim, click_num)
        # self.clk_subcate_co_gru = Sequence(self.dim, self.dim, click_num)

        # self.clk_cate_ti_en_gru = Sequence(self.dim, self.dim, click_num)
        # self.clk_cate_co_en_gru = Sequence(self.dim, self.dim, click_num)
        # self.clk_subcate_ti_en_gru = Sequence(self.dim, self.dim, click_num)
        # self.clk_subcate_co_en_gru = Sequence(self.dim, self.dim, click_num)

        self.ucate_part_fc = nn.Sequential(
#             nn.Dropout(dropout),
            nn.Linear((dim + word_dim * 2 + en_dim) + dim * 2, 512),
            # nn.LayerNorm([256]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, dim * 2)
        )

        self.usubcate_part_fc = nn.Sequential(
            nn.Linear(dim + word_dim * 2 + en_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, dim * 2)
        )

        self.uemb_fc = nn.Sequential(
            nn.Linear(dim * 2 * 2 + en_dim + word_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, final_dim)
        )

        self.agg_clk_ti_part_v1 = AggreAttention(word_dim, 512)
        self.agg_ti_en_part_v1 = AggreAttention(en_dim, 512)
        self.agg_clk_cate_emb = AggreAttention(word_dim, 512)
        self.agg_clk_subcate_emb = AggreAttention(word_dim, 512)
        self.agg_ti_en_part_v2 = AggreAttention(en_dim, 512)
        self.agg_clk_ti_part_v2 = AggreAttention(word_dim, 512)
        self.agg_usubcate_part = AggreAttention(dim * 2, 512)
        self.agg_ucate_part = AggreAttention(dim * 2, 512)

    def forward(self, clk_ti_emb, clk_cate_emb, clk_subcate_emb, clk_ti_en_emb):
        ## mask
        # one = torch.ones_like(clk_ti)
        # clk_ti_mask = torch.where(clk_ti > 0, one, clk_ti)

        # one = torch.ones_like(clk_co)
        # clk_co_mask = torch.where(clk_co > 0, one, clk_co)

        # one = torch.ones_like(clk_cate)
        # clk_cate_mask = torch.where(clk_cate > 0, one, clk_cate)

        # one = torch.ones_like(clk_subcate)
        # clk_subcate_mask = torch.where(clk_subcate > 0, one, clk_subcate)

        # one = torch.ones_like(clk_ti_en)
        # clk_ti_en_mask = torch.where(clk_ti_en > 0, one, clk_ti_en)

        # one = torch.ones_like(clk_co_en)
        # clk_co_en_mask = torch.where(clk_co_en > 0, one, clk_co_en)

        ## embedding
        # clk_ti_emb = self.embed_dropout(self.word_embedding(clk_ti))
        # clk_cate_emb = self.embed_dropout(self.cate_embedding(clk_cate))
        # clk_subcate_emb = self.embed_dropout(self.subcate_embedding(clk_subcate))
        # clk_ti_en_emb = self.embed_dropout(self.en_embedding(clk_ti_en))
        
        # 学习词向量，能够学到cross-word information
        clk_ti_a, clk_ti_b, clk_ti_c, clk_ti_d = clk_ti_emb.shape
        clk_ti_emb = clk_ti_emb.reshape(-1, clk_ti_c, clk_ti_d)
        # _clk_ti_mask = clk_ti_mask.view(-1, clk_ti_mask.shape[-1])
        clk_ti_emb = self.ti_bigru(clk_ti_emb)
        clk_ti_emb = clk_ti_emb.reshape(clk_ti_a, clk_ti_b, clk_ti_c, clk_ti_d) 

        # clk_ti_part = torch.mean(clk_ti_emb, dim=-2)
        clk_ti_part = self.agg_clk_ti_part_v1(clk_ti_emb).squeeze(-2)
        clk_ti_part = self.ti_news_bigru(clk_ti_part)
        # clk_co_a, clk_co_b, clk_co_c, clk_co_d = clk_co_emb.shape
        # clk_co_emb = clk_co_emb.view(-1, clk_co_c, clk_co_d)
        # _clk_co_mask = clk_co_mask.view(-1, clk_co_mask.shape[-1])
        # clk_co_emb = self.co_bigru(clk_co_emb, _clk_co_mask)
        # clk_co_emb = clk_co_emb.view(clk_co_a, clk_co_b, clk_co_c, clk_co_d)

        clk_ti_en_a, clk_ti_en_b, clk_ti_en_c, clk_ti_en_d = clk_ti_en_emb.shape
        clk_ti_en_emb = clk_ti_en_emb.reshape(-1, clk_ti_en_c, clk_ti_en_d)
        # _clk_ti_en_mask = clk_ti_en_mask.view(-1, clk_ti_en_mask.shape[-1])
        clk_ti_en_emb = self.ti_en_bigru(clk_ti_en_emb)
        clk_ti_en_emb = clk_ti_en_emb.reshape(clk_ti_en_a, clk_ti_en_b, clk_ti_en_c, clk_ti_en_d)

        # clk_co_en_a, clk_co_en_b, clk_co_en_c, clk_co_en_d = clk_co_en_emb.shape
        # clk_co_en_emb = clk_co_en_emb.view(-1, clk_co_en_c, clk_co_en_d)
        # _clk_co_en_mask = clk_co_en_mask.view(-1, clk_co_en_mask.shape[-1])
        # clk_co_en_emb = self.co_en_bigru(clk_co_en_emb, _clk_co_en_mask)
        # clk_co_en_emb = clk_co_en_emb.view(clk_co_en_a, clk_co_en_b, clk_co_en_c, clk_co_en_d)

        # ti_en_part = torch.mean(clk_ti_en_emb, dim=-2)
        ti_en_part = self.agg_ti_en_part_v1(clk_ti_en_emb).squeeze(-2)
        ti_en_part = self.ti_news_en_bigru(ti_en_part)
        # co_en_part = torch.sum(clk_co_en_emb, dim=-2)

        # cate和subcate时间序列建模
        clk_cate_emb = self.clk_cate_bigru(clk_cate_emb)

        clk_subcate_emb = self.clk_subcate_bigru(clk_subcate_emb)

        # subcate attention 挑词
        _clk_subcate_emb = clk_subcate_emb.unsqueeze(-2)
        # _clk_subcate_mask = clk_subcate_mask.unsqueeze(-1)
        ## title
        subcate_ti_part = self.subcate_ti_attr(_clk_subcate_emb, clk_ti_emb, clk_ti_emb).squeeze(-2)
        # subcate_ti_part = self.clk_subcate_ti_gru(subcate_ti_part, clk_subcate_mask)
        # subcate_ti_part = torch.mul(_clk_subcate_mask, subcate_ti_part)
        ### 合并多个文档
        

        ## content
        # subcate_co_part = self.subcate_co_attr(_clk_subcate_emb, clk_co_emb, clk_co_emb, clk_co_mask)
        # subcate_co_part = torch.mul(_clk_subcate_mask, subcate_co_part)
        ### 合并多个文档
        

        ## en
        # subcate_ti_en_part = self.subcate_ti_en_attr(_clk_subcate_emb, clk_ti_en_emb, clk_ti_en_emb, clk_ti_en_mask)
        # subcate_ti_en_part = torch.mul(_clk_subcate_mask, subcate_ti_en_part)
        # subcate_ti_en_part, _ = torch.max(clk_ti_en_emb, dim=-2)
        # _clk_subcate_emb = clk_ti_en_emb.unsqueeze(-2)
        # ti_en_part = self.ti_en_self_attr(_clk_subcate_emb, clk_ti_en_emb, clk_ti_en_emb, clk_ti_en_mask)
        # subcate_co_en_part = self.subcate_co_en_attr(_clk_subcate_emb, clk_co_en_emb, clk_co_en_emb, clk_co_en_mask)
        # subcate_co_en_part = torch.mul(_clk_subcate_mask, subcate_co_en_part)
        # subcate_co_en_part, _ = torch.max(clk_co_en_emb, dim=-2)

        usubcate_cat_part = torch.cat([subcate_ti_part, clk_subcate_emb, clk_ti_part, ti_en_part], dim=-1)
        usubcate_part = self.usubcate_part_fc(usubcate_cat_part)
        

        # cate attention 挑词
        _clk_cate_emb = clk_cate_emb.unsqueeze(-2)
        # _clk_cate_mask = clk_cate_mask.unsqueeze(-1)
        ## title
        cate_ti_part = self.cate_ti_attr(_clk_cate_emb, clk_ti_emb, clk_ti_emb).squeeze(-2)
        # cate_ti_part = torch.mul(_clk_cate_mask, cate_ti_part)

        ## content
        # cate_co_part = self.cate_co_attr(_clk_cate_emb, clk_co_emb, clk_co_emb, clk_co_mask)
        # cate_co_part = torch.mul(_clk_cate_mask, cate_co_part)

        ## en
        # cate_ti_en_part = self.cate_ti_en_attr(_clk_cate_emb, clk_ti_en_emb, clk_ti_en_emb, clk_ti_en_mask)
        # cate_ti_en_part = torch.mul(_clk_cate_mask, cate_ti_en_part)
        # cate_ti_en_part, _ = torch.max(clk_ti_en_emb, dim=-2)
        # cate_co_en_part = self.cate_co_en_attr(_clk_cate_emb, clk_co_en_emb, clk_co_en_emb, clk_co_en_mask)
        # cate_co_en_part = torch.mul(_clk_cate_mask, cate_co_en_part)
        # cate_co_en_part, _ = torch.max(clk_co_en_emb, dim=-2)

        ucate_cat_part = torch.cat([cate_ti_part, clk_cate_emb, clk_ti_part, ti_en_part], dim=-1)
        ucate_cat_part = torch.cat([ucate_cat_part, usubcate_part], dim=-1)
        ucate_part = self.ucate_part_fc(ucate_cat_part)

        # 合并cate和subcate
        # clk_cate_emb = torch.mean(clk_cate_emb, dim=-2, keepdim=True)
        # clk_subcate_emb = torch.mean(clk_subcate_emb, dim=-2, keepdim=True)
        # ti_en_part = torch.mean(ti_en_part, dim=-2, keepdim=True)
        # clk_ti_part = torch.mean(clk_ti_part, dim=-2, keepdim=True)
        # usubcate_part = torch.mean(usubcate_part, dim=-2, keepdim=True)
        # ucate_part = torch.mean(ucate_part, dim=-2, keepdim=True)

        clk_cate_emb = self.agg_clk_cate_emb(clk_cate_emb)
        clk_subcate_emb = self.agg_clk_subcate_emb(clk_subcate_emb)
        ti_en_part = self.agg_ti_en_part_v2(ti_en_part)
        clk_ti_part = self.agg_clk_ti_part_v2(clk_ti_part)
        usubcate_part = self.agg_usubcate_part(usubcate_part)
        ucate_part = self.agg_ucate_part(ucate_part)

        uemb_cat = torch.cat([ucate_part, usubcate_part, clk_ti_part, ti_en_part, clk_cate_emb, clk_subcate_emb], dim=-1)

        uemb = self.uemb_fc(uemb_cat)

        # uemb_normalized = F.normalize(uemb, p=2, dim=-1) / tao

        return uemb, uemb_cat, ucate_part, usubcate_part


class NewsEncoder(nn.Module):
    def __init__(self, ti_max_len, en_max_len) -> None:
        super(NewsEncoder, self).__init__()
        
        # self.embed_dropout = nn.Dropout(p=dropout)

        # self.cate_embedding = nn.Embedding.from_pretrained(cate_embeddings, freeze=False)
        # self.subcate_embedding = nn.Embedding.from_pretrained(subcate_embeddings, freeze=False)
        # self.word_embedding = nn.Embedding.from_pretrained(w2v, freeze=False)
        # self.en_embedding = nn.Embedding.from_pretrained(en_embeddings, freeze=False)
        # self.dropout = nn.Dropout(dropout)

        self.ti_bigru = Sequence(word_dim, word_dim, ti_max_len)
        # self.co_bigru = Sequence(self.word_dim, self.word_dim, co_max_len)
        self.ti_en_bigru = Sequence(en_dim, en_dim, en_max_len)
        # self.co_en_bigru = Sequence(self.en_dim, self.en_dim, en_max_len)

        self.cate_ti_attr = Attention(word_dim, word_dim, word_dim, hidden_dim=256, output_dim=dim)
        self.subcate_ti_attr = Attention(word_dim, word_dim, word_dim, hidden_dim=256, output_dim=dim)

        # self.cate_co_attr = Attention(self.word_dim, self.word_dim, self.word_dim, self.dim)
        # self.subcate_co_attr = Attention(self.word_dim, self.word_dim, self.word_dim, self.dim)

        # self.ti_en_attr = Attention(self.en_dim, self.en_dim, self.en_dim, self.dim)
        # self.co_en_attr = Attention(self.en_dim, self.en_dim, self.en_dim, self.dim)

        # self.cate_co_en_attr = Attention(self.word_dim, self.en_dim, self.en_dim, self.dim)
        # self.subcate_co_en_attr = Attention(self.word_dim, self.en_dim, self.en_dim, self.dim)

        self.icate_part_fc = nn.Sequential(
            nn.Linear((dim + en_dim + word_dim * 2) + dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim * 2)
        )

        self.isubcate_part_fc = nn.Sequential(
            nn.Linear(dim + en_dim + word_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim * 2)
        )

        self.iemb_fc = nn.Sequential(
            nn.Linear(dim * 2 * 2 + en_dim + 3 * word_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, final_dim)
        )

        self.agg_ti_part = AggreAttention(word_dim, 256)
        self.agg_ti_en_part = AggreAttention(en_dim, 256)
    
    def forward(self, ti_emb, cate_emb, subcate_emb, ti_en_emb):
        ## mask
        # one = torch.ones_like(ti)
        # ti_mask = torch.where(ti > 0, one, ti)

        # one = torch.ones_like(co)
        # co_mask = torch.where(co > 0, one, co)

        # one = torch.ones_like(ti_en)
        # ti_en_mask = torch.where(ti_en > 0, one, ti_en)

        # one = torch.ones_like(co_en)
        # co_en_mask = torch.where(co_en > 0, one, co_en)

        # ti_emb = self.embed_dropout(self.word_embedding(ti))
        # cate_emb = self.embed_dropout(self.cate_embedding(cate))
        # subcate_emb = self.embed_dropout(self.subcate_embedding(subcate))
        # ti_en_emb = self.embed_dropout(self.en_embedding(ti_en))

        # 序列关系，可能会慢
        ti_a, ti_b, ti_c, ti_d = ti_emb.shape
        ti_emb = ti_emb.view(-1, ti_c, ti_d)
        # _ti_mask = ti_mask.view(-1, ti_mask.shape[-1])
        ti_emb = self.ti_bigru(ti_emb)
        ti_emb = ti_emb.view(ti_a, ti_b, ti_c, ti_d)

        # ti_part = torch.mean(ti_emb, dim=-2)
        ti_part = self.agg_ti_part(ti_emb).squeeze(-2)

        # co_a, co_b, co_c, co_d = co_emb.shape
        # co_emb = co_emb.view(-1, co_c, co_d)
        # _co_mask = co_mask.view(-1, co_mask.shape[-1])
        # co_emb = self.co_bigru(co_emb, _co_mask)
        # co_emb = co_emb.view(co_a, co_b, co_c, co_d)
        
        co_a, co_b, co_c, co_d = ti_en_emb.shape
        ti_en_emb = ti_en_emb.view(-1, co_c, co_d)
        # _ti_en_mask = ti_en_mask.view(-1, ti_en_mask.shape[-1])
        ti_en_emb = self.ti_en_bigru(ti_en_emb)
        ti_en_emb = ti_en_emb.view(co_a, co_b, co_c, co_d)

        # co_a, co_b, co_c, co_d = co_en_emb.shape
        # co_en_emb = co_en_emb.view(-1, co_c, co_d)
        # _co_en_mask = co_en_mask.view(-1, co_en_mask.shape[-1])
        # co_en_emb = self.co_en_bigru(co_en_emb, _co_en_mask)
        # co_en_emb = co_en_emb.view(co_a, co_b, co_c, co_d)

        # ti_en_part = torch.mean(ti_en_emb, dim=-2)
        ti_en_part = self.agg_ti_en_part(ti_en_emb).squeeze(-2)
        # co_en_part, _ = torch.max(co_en_emb, dim=-2)

        # subcate attention 挑词
        _subcate_emb = subcate_emb.unsqueeze(-2)
        ## title
        isubcate_ti_part = self.subcate_ti_attr(_subcate_emb, ti_emb, ti_emb).squeeze(-2)
        ## content
        # isubcate_co_part = self.subcate_co_attr(_subcate_emb, co_emb, co_emb, co_mask)
        ## en
        # isubcate_ti_en_part = self.subcate_ti_en_attr(_subcate_emb, ti_en_emb, ti_en_emb, ti_en_mask)
        # isubcate_ti_en_part, _ = torch.max(ti_en_emb, dim=-2)
        # isubcate_co_en_part = self.subcate_co_en_attr(_subcate_emb, co_en_emb, co_en_emb, co_en_mask)
        # isubcate_co_en_part, _ = torch.max(co_en_emb, dim=-2)

        # 合并
        isubcate_cat_part = torch.cat([isubcate_ti_part, subcate_emb, ti_part, ti_en_part], dim=-1)
        isubcate_part = self.isubcate_part_fc(isubcate_cat_part)


        # cate attention 挑词
        _cate_emb = cate_emb.unsqueeze(-2)
        ## title
        icate_ti_part = self.cate_ti_attr(_cate_emb, ti_emb, ti_emb).squeeze(-2)
        ## cotent
        # icate_co_part = self.cate_co_attr(_cate_emb, co_emb, co_emb, co_mask)
        ## en
        # icate_ti_en_part = self.cate_ti_en_attr(_cate_emb, ti_en_emb, ti_en_emb, ti_en_mask)
        # icate_ti_en_part, _ = torch.max(ti_en_emb, dim=-2)
        # icate_co_en_part = self.cate_co_en_attr(_cate_emb, co_en_emb, co_en_emb, co_en_mask)
        # icate_co_en_part, _ = torch.max(co_en_emb, dim=-2)

        # cate_emb, _ = torch.max(cate_emb, dim=-2)
        # subcate_emb, _ = torch.max(subcate_emb, dim=-2)

        ## 合并
        icate_cat_part = torch.cat([icate_ti_part, cate_emb, ti_part, ti_en_part], dim=-1)
        icate_cat_part = torch.cat([icate_cat_part, isubcate_part], dim=-1)
        icate_part = self.icate_part_fc(icate_cat_part)

        iemb_cat = torch.cat([icate_part, isubcate_part, ti_part, ti_en_part, cate_emb, subcate_emb], dim=-1)
        iemb = self.iemb_fc(iemb_cat)

        # iemb_normalized = F.normalize(iemb, p=2, dim=-1) / tao

        return iemb, iemb_cat, icate_part, isubcate_part


class Attention(nn.Module):
    def __init__(self, q_input_dim, k_input_dim, v_input_dim, hidden_dim=256, output_dim=256) -> None:
        super(Attention, self).__init__()
        self.sqrt_dim = output_dim ** 0.5
        self.Q = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.K = nn.Sequential(
            nn.Linear(k_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.V = nn.Sequential(
            nn.Linear(v_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.att_dropout = nn.Dropout(dropout)

    def forward(self, q_x, k_x, v_x, mask=None):
        q_x = self.Q(q_x)
        k_x = self.K(k_x)
        v_x = self.V(v_x)
        
        q_x = torch.transpose(q_x, -1, -2)
        t = torch.matmul(k_x, q_x) / self.sqrt_dim
        # if mask is not None:
        #     mask = mask.unsqueeze(-1)
        #     t = t.masked_fill(mask==0, -1e9)
        weight = torch.softmax(t, dim=-2)
        if dropout is not None:
            weight = self.att_dropout(weight)
        weight = torch.transpose(weight, -1, -2)
        t = torch.matmul(weight, v_x)
        t = self.fc(t)
        return t


class AggreAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(AggreAttention, self).__init__()
        self.q = nn.Parameter(torch.zeros(hidden_dim, 1))
        # nn.init.kaiming_normal_(self.q)
        # self.q_fc = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU()
        # )
        # nn.init.kaiming_normal_(self.q, mode='fan_out', nonlinearity='relu')
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        x_part = self.fc(x)
        w = torch.matmul(x_part, self.q)
        att_w = torch.softmax(w, dim=-2)
        val = torch.matmul(att_w.transpose(-1, -2), x)
        return val


class Sequence(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm, layer_num=2) -> None:
        super(Sequence, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, activation="gelu", batch_first=True)
        self.att = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self.att = nn.GRU(input_dim, input_dim, 2, bidirectional=True)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     # nn.LayerNorm([512]),
        #     nn.GELU(),
        #     # nn.BatchNorm1d(1024),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, 256),
        #     # nn.LayerNorm([256]),
        #     nn.GELU(),
        #     # nn.BatchNorm1d(512),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, output_dim)
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(input_dim * 2, output_dim)
        # )

    def forward(self, x, mask=None):
        emb = self.att(x)
        # emb, _ = self.att(x)
        return emb
