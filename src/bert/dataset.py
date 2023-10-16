import torch
from torch.utils.data import Dataset
import numpy as np


class MindTrainDataset(Dataset):
    def __init__(self, user_idx, clk_news_idx, clk_ti, clk_cate, 
        clk_subcate, clk_ti_en, news_idx, ti, cate, subcate, ti_en):
        super().__init__()
        self.user_idx = user_idx

        self.clk_news_idx = clk_news_idx
        self.clk_ti = clk_ti
        self.clk_cate = clk_cate
        self.clk_subcate = clk_subcate
        self.clk_ti_en = clk_ti_en

        self.news_idx = news_idx
        self.ti = ti
        self.cate = cate
        self.subcate = subcate
        self.ti_en = ti_en

    def __getitem__(self, index: int):
        return torch.tensor(self.user_idx[index]), \
            torch.tensor(self.clk_news_idx[index]), \
            torch.tensor(np.array(self.clk_ti[index])), \
            torch.tensor(self.clk_cate[index]), \
            torch.tensor(self.clk_subcate[index]), \
            torch.tensor(self.clk_ti_en[index]), \
            torch.tensor(self.news_idx[index]), \
            torch.tensor(np.array(self.ti[index])), \
            torch.tensor(self.cate[index]), \
            torch.tensor(self.subcate[index]), \
            torch.tensor(self.ti_en[index])
            # torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.user_idx)


class MindUserDataset(Dataset):
    def __init__(self, sess_idx, clk_ti, clk_cate, clk_subcate,
        clk_ti_en) -> None:
        self.sess_idx = sess_idx

        self.clk_ti = clk_ti
        self.clk_cate = clk_cate
        self.clk_subcate = clk_subcate
        self.clk_ti_en = clk_ti_en
    
    def __getitem__(self, index):
        return self.sess_idx[index], \
            torch.tensor(np.array(self.clk_ti[index])), \
            torch.tensor(self.clk_cate[index]), \
            torch.tensor(self.clk_subcate[index]), \
            torch.tensor(self.clk_ti_en[index])
    
    def __len__(self) -> int:
        return len(self.sess_idx)


class MindNewsDataset(Dataset):
    def __init__(self, news_idx, ti, cate, subcate, ti_en) -> None:
        self.news_idx = news_idx
        self.ti = ti
        self.cate = cate
        self.subcate = subcate
        self.ti_en = ti_en
    
    def __getitem__(self, index):

        return self.news_idx[index], \
            torch.tensor(np.array([self.ti[index]])), \
            torch.tensor([self.cate[index]]), \
            torch.tensor([self.subcate[index]]), \
            torch.tensor([self.ti_en[index]])
    
    def __len__(self) -> int:
        return len(self.ti)


class MindBertTrainDataset(Dataset):
    def __init__(self, user_idx, clk_news_idx, clk_cate, 
        clk_subcate, clk_ti_en, news_idx, cate, subcate, ti_en):
        super().__init__()
        self.user_idx = user_idx

        self.clk_news_idx = clk_news_idx
        self.clk_cate = clk_cate
        self.clk_subcate = clk_subcate
        self.clk_ti_en = clk_ti_en

        self.news_idx = news_idx
        self.cate = cate
        self.subcate = subcate
        self.ti_en = ti_en

    def __getitem__(self, index: int):
        return self.user_idx[index], \
            self.clk_news_idx[index], \
            torch.as_tensor(np.array(self.clk_cate[index])), \
            torch.as_tensor(np.array(self.clk_subcate[index])), \
            torch.as_tensor(np.array(self.clk_ti_en[index])), \
            self.news_idx[index], \
            torch.as_tensor(np.array(self.cate[index])), \
            torch.as_tensor(np.array(self.subcate[index])), \
            torch.as_tensor(np.array(self.ti_en[index]))

    def __len__(self) -> int:
        return len(self.user_idx)


class MindBertUserDataset(Dataset):
    def __init__(self, sess_idx, clk_news_idx, clk_cate, clk_subcate,
        clk_ti_en) -> None:
        self.sess_idx = sess_idx
        self.clk_news_idx = clk_news_idx

        self.clk_cate = clk_cate
        self.clk_subcate = clk_subcate
        self.clk_ti_en = clk_ti_en
    
    def __getitem__(self, index):
        return self.sess_idx[index], \
            self.clk_news_idx[index], \
            torch.as_tensor(np.array(self.clk_cate[index])), \
            torch.as_tensor(np.array(self.clk_subcate[index])), \
            torch.as_tensor(np.array(self.clk_ti_en[index]))
    
    def __len__(self) -> int:
        return len(self.sess_idx)


class MindBertNewsDataset(Dataset):
    def __init__(self, news_idx, cate, subcate, ti_en) -> None:
        self.news_idx = news_idx
        self.cate = cate
        self.subcate = subcate
        self.ti_en = ti_en
    
    def __getitem__(self, index):

        return self.news_idx[index], \
            torch.as_tensor(np.array([self.cate[index]])), \
            torch.as_tensor(np.array([self.subcate[index]])), \
            torch.as_tensor(np.array([self.ti_en[index]]))
    
    def __len__(self) -> int:
        return len(self.news_idx)