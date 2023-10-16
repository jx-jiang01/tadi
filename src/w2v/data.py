# -*- coding: utf-8 -*-
import random
import json
import numpy as np
import nltk
import scipy.stats as stats
from utils import cate_large_reset_dict as cate_reset_dict
from utils import subcate_large_reset_dict as subcate_reset_dict


# random.seed(21)
root = "/data/nlp/jiangjx/"
data_root = root + "data/news/mind-small/"
model_root = root + "model/news/"
vocab_path = root + "nlp/glove/glove.840B.300d.txt"
word_idx_path = root + "nlp/glove/word_idx.txt"


def truncated_normal(size, loc=0, scale=0.01):
    # X = stats.truncnorm(-2, 2, loc=0, scale=0.001)
    # val = X.rvs(size)
    return np.zeros(size)
    

def load_mind_impression_data(path, num=-1):
    session_ids, user_ids, history_clicks, impressions = [], [], [], []
    cnt = 0
    with open(path, encoding="utf-8") as f:
        line = f.readline()
        cnt += 1
        while line:
            if num > 0 and cnt > num:
                break
            terms = line.strip().split("\t")
            session_ids.append(int(terms[0]))
            user_id = terms[1].strip()
            if terms[3].strip() == "":
                history_click = []
            else:
                history_click = terms[3].strip().split(" ")
            impression = terms[4].strip().split(" ")
            user_ids.append(user_id)
            history_clicks.append(history_click)
            impressions.append(impression)
            line = f.readline()
            cnt += 1
        f.close()
    return session_ids, user_ids, history_clicks, impressions


def cate_embedding(vocab_embs, vocab_idx, data_dict):
    res_dict = {}
    for key in data_dict:
        words = data_dict[key]
        val = []
        for word in words:
            val.append(vocab_embs[vocab_idx[word]])
        val = np.array(val)
        max_v = np.max(val, axis=0)
        # mean_v = np.mean(val, axis=0)
        # res_dict[key] = (max_v + mean_v) / 2
        res_dict[key] = max_v
    return res_dict


def preprocess_cate_subcate(vocab_embs, vocab_idx, cate_path, subcate_path):
    cate_embs = cate_embedding(vocab_embs, vocab_idx, cate_reset_dict)
    subcate_embs = cate_embedding(vocab_embs, vocab_idx, subcate_reset_dict)

    with open(cate_path, "w") as f:
        for key in cate_embs:
            val = cate_embs[key]
            f.write(key + "\001" + " ".join([str(v) for v in val]) + "\n")
        f.close()
    with open(subcate_path, "w") as f:
        for key in subcate_embs:
            val = subcate_embs[key]
            f.write(key + "\001" + " ".join([str(v) for v in val]) + "\n")
        f.close()


def load_cate_embedding(path):
    cate_id_idx, cate_emb = {"[PAD]": 0}, [truncated_normal(300)]
    with open(path) as f:
        line = f.readline()
        while line:
            key, val = line.strip().split("\001")
            if key not in cate_id_idx:
                val = np.array([float(v) for v in val.split(" ")])
                cate_id_idx[key] = len(cate_id_idx)
                cate_emb.append(val)
            line = f.readline()
    return cate_id_idx, cate_emb


def load_entity_embedding(paths):
    entity_id_idx, entity_embeddings = {"[PAD]": 0}, [truncated_normal(100, scale=0.001)]
    for path in paths:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
            while line:
                terms = line.strip().split("\t")
                key = terms[0]
                val = np.array([float(v) for v in terms[1:]])
                if key not in entity_id_idx:
                    entity_id_idx[key] = len(entity_id_idx)
                    entity_embeddings.append(val)
                line = f.readline()
    return entity_id_idx, entity_embeddings


def load_mind_news_data(paths, cate_id_idx, subcate_id_idx, en_id_idx, ti_max_len, 
    en_max_len, word_idx=None):

    news_id_idx = {"[PAD]": 0}
    news_ti = {"[PAD]": [word_idx["[PAD]"]] * ti_max_len}

    news_cate = {"[PAD]": cate_id_idx["[PAD]"]}
    news_subcate = {"[PAD]": subcate_id_idx["[PAD]"]}
    news_ti_en = {"[PAD]": [en_id_idx["[PAD]"]] * en_max_len}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
            while line:
                terms = line.strip().split("\t")
                news_id, category, subcategory, title = terms[: 4]

                line = f.readline()
                if news_id in news_id_idx:
                    continue

                ti_entity = json.loads(terms[6])

                ti_en = []
                for _entity in ti_entity:
                    en_id = _entity["WikidataId"]
                    offsets = _entity["OccurrenceOffsets"]
                    if en_id in en_id_idx:
                        # ti_en.append([en_id_idx[en_id]])
                        for offset in offsets:
                            ti_en.append([en_id_idx[en_id], offset])

                ti_en.sort(key=lambda x: x[1])
                
                ti_en_idx = []
                for _en in ti_en:
                    ti_en_idx.append(_en[0])
                if len(ti_en_idx) > en_max_len:
                    ti_en_idx = ti_en_idx[: en_max_len]
                else:
                    ti_en_idx += [en_id_idx["[PAD]"]] * (en_max_len - len(ti_en_idx))
                
                news_ti[news_id] = transform(word_idx, [title], ti_max_len)[0]
                
                news_id_idx[news_id] = len(news_id_idx)
                news_cate[news_id] = cate_id_idx[category]
                news_subcate[news_id] = subcate_id_idx[subcategory]
                news_ti_en[news_id] = ti_en_idx
                
            f.close()
    return news_id_idx, news_ti, news_cate, news_subcate, news_ti_en


# stop_words = set(stopwords.words('english'))


def word_splitter(text):
    # text.lower().split(" ")
    # nltk.word_tokenize(text)
    words = nltk.word_tokenize(text)
    filtered = []
    for w in words:
        filtered.append(w)
        # if w not in stop_words and w.isalpha():
        #     filtered.append(w)
    return filtered


def transform(word_idx, texts, maximum=None):
    word_vectors = []  # blank的特征
    for text in texts:
        word_vector = []
        words = word_splitter(text)  # 分词要修改
        for word in words:
            if word in word_idx:
                word_vector.append(word_idx[word])
            # else:
            #     word_vector.append(word_idx["[blank]"])
        if maximum is not None:
            diff = maximum - len(word_vector)
            if diff >= 0:
                word_vector += [word_idx["[PAD]"]] * diff
            else:
                word_vector = word_vector[: maximum]
        word_vectors.append(word_vector)
    return word_vectors


def category_transform(categories, cate_id_idx=None):
    _categories = []  # 第一个表示查不到category 
    if cate_id_idx is None:
        cate_id_idx = {"[blank]": 0}
        for category in categories:
            if category not in cate_id_idx:
                cate_id_idx[category] = len(cate_id_idx)
            _categories.append(cate_id_idx[category])
    else:
        for category in categories:
            if category in cate_id_idx:
                _categories.append(cate_id_idx[category])
            else:
                _categories.append(cate_id_idx["[blank]"])
    return _categories, cate_id_idx


def id_transform(ids, id_idx=None):
    _idx = []  # 第一个表示找不到id
    if id_idx is None:
        id_idx = {"[blank]": 0}
        for id in ids:
            if id not in id_idx:
                id_idx[id] = len(id_idx)
            _idx.append(id_idx[id])
    else:
        for id in ids:
            if id in id_idx:
                _idx.append(id_idx[id])
            else:
                _idx.append(id_idx["[blank]"])
    return _idx, id_idx


def load_vocab_embs(w2v_path, padding=False):
    vocab_idx, vocab_embs = {}, []
    if padding:
        vocab_idx, vocab_embs = {"[PAD]": 0}, [truncated_normal(300)]

    with open(w2v_path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            terms = line.strip().split(" ")
            key, val = terms[0], np.array([float(v) for v in terms[1:]], dtype=np.float32)
            vocab_idx[key] = len(vocab_idx)
            vocab_embs.append(val)
            line = f.readline()
        f.close()
    return vocab_embs, vocab_idx


def filter_vocab_embedding(vocab_embs, vocab_idx, filtered_paths):
    word_embs = []
    word_idx = {}  # 用[blank]来表示padding的单词
    word_set = set()
    for path in filtered_paths:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
            while line:
                terms = line.strip().split("\t")
                title = terms[3]
                words = word_splitter(title)
                for word in words:
                    if word not in word_set:
                        word_set.add(word)
                line = f.readline()
            f.close()
    for key in vocab_idx:
        if key in word_set and key not in word_idx:
            val = vocab_embs[vocab_idx[key]]
            word_idx[key] = len(word_idx)
            word_embs.append(val)
    return word_embs, word_idx


def save_w2v(w2v_path, word_embs, word_idx):
    word_idx = sorted(word_idx.items(), key=lambda x: x[1])
    with open(w2v_path, "w") as f:
        for word, idx in word_idx:
            f.write(word + " " + " ".join([str(v) for v in word_embs[idx]]) + "\n")
        f.close()


def load_word_idx(word_idx_path):
    word_idx = {}
    with open(word_idx_path, encoding="utf-8") as f:
        line = f.readline()
        while line:
            terms = line.strip().split("\001")
            word_idx[terms[0]] = int(terms[1])
            line = f.readline()
        f.close()
    return word_idx


def one_sample_feature(sample_idx, title_vectors, content_vectors, categories, subcategories):
    # idx = id_idx[sample_id]

    _title = title_vectors[sample_idx]
    _category = categories[sample_idx]
    _subcategory = subcategories[sample_idx]
    _content = content_vectors[sample_idx]
    
    return _title, _content, _category, _subcategory


def one_group_sample_feature(idx, title_vectors, content_vectors, cate_idx, 
        subcate_idx):
    # 1个position搭配多个negative
    titles, contents, cate, subcate = [], [], [], []
    for _idx in idx:
        _title, _content, _cate, _subcate = one_sample_feature(_idx, 
            title_vectors, content_vectors, cate_idx, 
            subcate_idx)
        titles.append(_title)
        contents.append(_content)
        cate.append(_cate)
        subcate.append(_subcate)
    return titles, contents, cate, subcate


def load_dataset():
    pass


def negative_sampling(neg_news_ids, K, max_num=-1):
    
    if len(neg_news_ids) <= K:
        all = neg_news_ids * (K // len(neg_news_ids) + 1)
        selected_samples = [random.sample(all, K)]
    else:
        rest = len(neg_news_ids) % K
        additional = random.sample(neg_news_ids, K - rest)
        all = additional + neg_news_ids
        random.shuffle(all)
        selected_samples = [all[i: i + K] for i in range(0, len(all), K)]
    
    selected_samples = selected_samples[: max_num]
    return selected_samples


def _processed_data_pairs(K, session_ids, user_sess_idx, impressions, clk_sess_news_idx, clk_sess_ti, 
    clk_sess_cate, clk_sess_subcate, clk_sess_ti_en, news_id_idx, news_ti, 
    news_cate, news_subcate, news_ti_en, data_type):
    sess_ids = []
    user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate = [], [], [], [], []
    clk_ti_en = []
    news_idx = []
    sess_behaviors = []
    ti, ti_en = [], []
    cate, subcate = [], []
    labels = []

    if data_type == "test" or data_type == "dev":
        news_idx_set = set()
        
        for _session_id, _user_idx, _impression, _clk_sess_news_idx, _clk_sess_ti, _clk_sess_cate, \
            _clk_sess_subcate, _clk_sess_ti_en in zip(session_ids, user_sess_idx, impressions, clk_sess_news_idx, 
                clk_sess_ti, clk_sess_cate, clk_sess_subcate, clk_sess_ti_en):
            _news_idx = []
            _labels = []
            for impress in _impression:
                if data_type == "dev":
                    news_id, label = impress.split("-")
                    _cur_news_idx = news_id_idx[news_id]

                    _labels.append(int(label))
                else:
                    news_id = impress
                    _cur_news_idx = news_id_idx[news_id]

                _news_idx.append(_cur_news_idx)
                
                # 用于news encoding
                if _cur_news_idx not in news_idx_set:
                    news_idx_set.add(_cur_news_idx)
                    news_idx.append(_cur_news_idx)
                    ti.append(news_ti[news_id])
                    cate.append(news_cate[news_id])
                    subcate.append(news_subcate[news_id])
                    ti_en.append(news_ti_en[news_id])
            
            # 用于user encoding
            sess_ids.append(_session_id)
            user_idx.append(_user_idx)
            clk_news_idx.append(_clk_sess_news_idx)
            clk_ti.append(_clk_sess_ti)
            clk_cate.append(_clk_sess_cate)
            clk_subcate.append(_clk_sess_subcate)
            clk_ti_en.append(_clk_sess_ti_en)

            sess_behavior = []
            sess_behavior.append(_session_id)
            sess_behavior.append(_news_idx)
            sess_behavior.append(_labels)
            sess_behaviors.append(sess_behavior)

        return sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, sess_behaviors

    for _session_ids, _user_idx, _impression, _clk_sess_news_idx, _clk_sess_ti, _clk_sess_cate, \
        _clk_sess_subcate, _clk_sess_ti_en in zip(session_ids, user_sess_idx, impressions, clk_sess_news_idx, 
            clk_sess_ti, clk_sess_cate, clk_sess_subcate, clk_sess_ti_en):

        pos_news_ids, neg_news_ids = [], []
        for impress in _impression:
            news_id, label = impress.split("-")
            
            if label == "1":
                pos_news_ids.append(news_id)
            else:
                neg_news_ids.append(news_id)
        
        for pos_news_id in pos_news_ids:
            all_selected_samples = negative_sampling(neg_news_ids, K, 1)
            for selected_samples in all_selected_samples:
                _news_idx, _ti, _cate, _subcate, _ti_en = [], [], [], [], []

                sess_ids.append(_session_ids)
                user_idx.append(_user_idx)
                clk_news_idx.append(_clk_sess_news_idx)
                clk_ti.append(_clk_sess_ti)
                clk_cate.append(_clk_sess_cate)
                clk_subcate.append(_clk_sess_subcate)
                clk_ti_en.append(_clk_sess_ti_en)

                selected_samples.append(pos_news_id)

                for n_id in selected_samples:
                    _news_idx.append(news_id_idx[n_id])
                    _ti.append(news_ti[n_id])
                    _cate.append(news_cate[n_id])
                    _subcate.append(news_subcate[n_id])
                    _ti_en.append(news_ti_en[n_id])
                news_idx.append(_news_idx)
                ti.append(_ti)
                cate.append(_cate)
                subcate.append(_subcate)
                ti_en.append(_ti_en)

    return sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
        news_idx, ti, cate, subcate, ti_en, labels


def processed_data(behavior_path, K, click_num, ti_max_len, en_max_len,
    news_id_idx, news_ti, news_cate, news_subcate, news_ti_en, user_id_idx=None, 
    data_type="train", num=-1):

    session_ids, user_ids, clicks, impressions = load_mind_impression_data(behavior_path, num)
    
    if data_type=="train":
        user_sess_idx, user_id_idx = id_transform(user_ids)
    else:
        user_sess_idx, _ = id_transform(user_ids, user_id_idx)

    print("data loaded")
    
    # processed data

    clk_sess_ti, clk_sess_cate, clk_sess_subcate, clk_sess_ti_en = [], [], [], []
    clk_sess_news_idx = []
    for _clicks in clicks:
        _clicks = _clicks[-click_num: ]
        ti_v, cate_v, subcate_v, ti_en_v = [], [], [], []
        
        _click_idx = []
        for _click in _clicks:
            idx = news_id_idx[_click]
            _click_idx.append(idx)
            ti_v.append(news_ti[_click])
            cate_v.append(news_cate[_click])
            subcate_v.append(news_subcate[_click])
            ti_en_v.append(news_ti_en[_click])
        diff = click_num - len(_clicks)
        if diff > 0:
            ti_v = diff * [news_ti["[PAD]"]] + ti_v
            ti_en_v = diff * [news_ti_en["[PAD]"]] + ti_en_v
            _click_idx = diff * [0] + _click_idx
            cate_v = diff * [news_cate["[PAD]"]] + cate_v
            subcate_v = diff * [news_subcate["[PAD]"]] + subcate_v
        clk_sess_news_idx.append(_click_idx)
        clk_sess_ti.append(ti_v)
        clk_sess_cate.append(cate_v)
        clk_sess_subcate.append(subcate_v)
        clk_sess_ti_en.append(ti_en_v)
    if data_type == "train":
        sess_idx, \
        user_idx, clk_news_ids, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
        news_idx, ti, cate, subcate, ti_en, labels = _processed_data_pairs(K, session_ids, user_sess_idx, \
            impressions, clk_sess_news_idx, clk_sess_ti, clk_sess_cate, clk_sess_subcate, \
                clk_sess_ti_en, news_id_idx, \
                news_ti, news_cate, news_subcate, news_ti_en, data_type)

        return user_id_idx, \
            sess_idx, user_idx, clk_news_ids, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
                news_idx, ti, cate, subcate, ti_en, labels

    else:
        sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, sess_behaviors = _processed_data_pairs(K, session_ids, user_sess_idx, \
                impressions, clk_sess_news_idx, clk_sess_ti, clk_sess_cate, \
                    clk_sess_subcate, clk_sess_ti_en, news_id_idx, news_ti, news_cate, news_subcate, \
                        news_ti_en, data_type)
        return sess_ids, user_idx, clk_news_idx, clk_ti, clk_cate, clk_subcate, clk_ti_en, \
            news_idx, ti, cate, subcate, ti_en, sess_behaviors


def load_ti_bert_embs(path):
    ti_embs = {}
    with open(path) as f:
        line = f.readline()
        while line:
            terms = line.split("\001")
            key, val = terms[0], np.array([np.float32(v) for v in terms[1].split(" ")])
            ti_embs[key] = [val]
            line = f.readline()
        f.close()
    return ti_embs


def load_cate_subcate(paths):
    cate_exist_keys = set(cate_reset_dict.keys())
    subcate_exist_keys = set(subcate_reset_dict.keys())
    cate_set, subcate_set = set(), set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
            while line:
                terms = line.strip().split("\t")
                _, cate, subcate, _, _ = terms[: 5]
                if cate not in cate_set:
                    cate_set.add(cate)
                if subcate not in subcate_set:
                    subcate_set.add(subcate)
                line = f.readline()
            f.close()
    print(len(cate_set.difference(cate_exist_keys)))
    subcate_diff = subcate_set.difference(subcate_exist_keys)
    print(len(subcate_diff))
    print("\n".join(subcate_diff))


if __name__ == "__main__":
    data_root = "/root/autodl-tmp/data/news/mind-large/"
    news_paths = [data_root + "train/news.tsv", data_root + "dev/news.tsv", data_root + "test/news.tsv"]
    cate_set = set()
    news_cate, news_subcate = {}, {}
    for path in news_paths:
        with open(path, encoding="utf-8") as f:
            line = f.readline()
            while line:
                terms = line.strip().split("\t")
                news_id, category, subcategory, title = terms[: 4]
                news_cate[news_id] = category
                news_subcate[news_id] = subcategory
                if category not in cate_set:
                    cate_set.add(category)
                line = f.readline()
    print(cate_set)
    train_beh_path = data_root + "train/behaviors.tsv"
    # test_beh_path = data_root + "test/behaviors.tsv"
    dev_beh_path = data_root + "dev/behaviors.tsv"

    train_cate, train_subcate = set(), set()
    session_ids, user_ids, clicks, impressions = load_mind_impression_data(train_beh_path)
    for _clicks in clicks:
        for _click in _clicks:
            train_cate.add(news_cate[_click])
            train_subcate.add(news_subcate[_click])
    
    from utils import cate_reset_dict, subcate_reset_dict

    print("cate")
    for cate in train_cate:
        print("'" + cate + "': " + str(cate_reset_dict[cate]))

    print("subcate")
    for subcate in train_subcate:
        if subcate not in subcate_reset_dict:
            print("'" + subcate + "': ")
        else:
            print("'" + subcate + "': " + str(subcate_reset_dict[subcate]))

    # test_cate, test_subcate = set(), set()
    # session_ids, user_ids, clicks, impressions = load_mind_impression_data(test_beh_path)
    # for _clicks in clicks:
    #     for _click in _clicks:
    #         if news_cate[_click] not in train_cate:
    #             test_cate.add(news_cate[_click])
    #         if news_subcate[_click] not in train_subcate:
    #             test_subcate.add(news_subcate[_click])
    # print("test_cate", test_cate)
    # print("test_subcate", test_subcate)

    dev_cate, dev_subcate = set(), set()
    session_ids, user_ids, clicks, impressions = load_mind_impression_data(dev_beh_path)
    for _clicks in clicks:
        for _click in _clicks:
            if news_cate[_click] not in train_cate:
                dev_cate.add(news_cate[_click])
            if news_subcate[_click] not in train_subcate:
                dev_subcate.add(news_subcate[_click])
    print("dev_cate", dev_cate)
    print("dev_cate", dev_subcate)

    # load_cate_subcate(news_paths)
    # import nltk
    # from nltk.corpus import stopwords
    # nltk.download('stopwords')
    # #nltk.download('punkt')
    # words = nltk.word_tokenize("it is a basketball.")
    # filtered = [w for w in words if w not in stopwords.words('english') and w.isalpha()]
    # print(words)
    # print(filtered)
    
    # word_idx = load_word_idx(word_idx_path)

    # from utils import cate_reset_dict, subcate_reset_dict
    # cnt = 0
    # for key in cate_reset_dict:
    #     cate = cate_reset_dict[key]
    #     for _cate in cate:
    #         if _cate not in word_idx:
    #             print("cate", _cate)
    # print()
    # for key in subcate_reset_dict:
    #     cate = subcate_reset_dict[key]
    #     for _cate in cate:
    #         if _cate not in word_idx:
    #             print("subcate", _cate)
    # print()
