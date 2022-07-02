import csv
from doctest import Example
from lib2to3.pgen2 import token
import os
import re
import collections
from nltk.corpus import stopwords
from nltk import word_tokenize,pos_tag
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l


DATASET_PATH = 'D:/Experiment/datasets'
ORIGIN_DATASET_PATH = 'D:/Experiment/origin_datasets/'

#datasets path
EXIST2021 = 'D:/Experiment/datasets/EXIST2021'
TRAIN_EXIST2021 = EXIST2021 + '/train.tsv'
TEST_EXIST2021 = EXIST2021 + '/test.tsv'
HASOC_2020 = 'D:/Experiment/datasets/HASOC2020'

EMBEDDING_PATH = 'D:/Experiment/wordembedding/'

class Pre_processing_tweets:

    def __init__(self):
        self.sr = stopwords.words('english')

    def clean_unuseful(self, text):
        """
        去除@user 和 url 等无用信息
        """
        # 去除Email
        text = re.sub(r'^([\w]+\.*)([\w]+)\@[\w]+\.\w{3}(\.\w{2}|)', ' ', text)
        # 去除url
        text = re.sub(r'^(https:\S+)', ' ', text)
        text = re.sub(r'[a-zA-Z]+://[^\s]*', '', text)
        # 去除@username 句柄
        text = re.sub(r'@[\w]*', '', text)
        # 去除特殊符号
        p_text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(text)
        text = "".join(p_text)
        text = re.sub(r'[\d+___|_]', '', text)

        return text

    def tokenize(self, sentence):
        """
        去除多余空白、分词、词性标注
        """
        try:
            sentence = re.sub(r'\s+', ' ', sentence)
            token_words = word_tokenize(sentence)
            #token_words = pos_tag(token_words)
        except:
            print(sentence)
            token_words = sentence.split()
        return token_words

    def is_number(self, s):
        """
        判断字符串是否为数字
        """
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def to_lower(self, token_words):
        """
        统一为小写
        """
        words_lists = [x.lower() for x in token_words]
        return words_lists

    def delete_stopwords(self, token_words):
        """
        去除停用词
        """
        cleaned_words = [word for word in token_words if word not in self.sr]
        return cleaned_words

    def stem(self, token_words):
        """
        词形归一化
        """
        words_lematizer = []
        wnl = WordNetLemmatizer()
        for word, tag in token_words:
            if tag.startswith('NN'):
                word_lematizer =  wnl.lemmatize(word, pos='n')  # n代表名词
            elif tag.startswith('VB'):
                word_lematizer =  wnl.lemmatize(word, pos='v')   # v代表动词
            elif tag.startswith('JJ'):
                word_lematizer =  wnl.lemmatize(word, pos='a')   # a代表形容词
            elif tag.startswith('R'):
                word_lematizer =  wnl.lemmatize(word, pos='r')   # r代表代词
            else:
                word_lematizer =  wnl.lemmatize(word)
            words_lematizer.append(word_lematizer)
        return words_lematizer

    def tokenize_process(self, texts):
        res = []
        for text in texts:
            token_words = self.tokenize(text)
            token_words = self.to_lower(token_words)
            token_words = self.stem(token_words)
            # token_words = self.delete_stopwords(token_words)
            res.append(token_words)
        return res


def count_corpus(tokens):
    """
    Count token frequencies.
    Defined in :numref:`sec_text_preprocessing`
    """
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """
    Vocabulary for text.
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def truncate_pad(line, num_steps, padding_token):
    """
    Truncate or pad sequences.
    Defined in :numref:`sec_machine_translation`
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def load_array(data_arrays, batch_size, is_train=True):
    """
    Construct a PyTorch data iterator.
    Defined in :numref:`sec_linear_concise`
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


"""
HASOC2020数据集加载函数
"""
def set_hasoc2020_data(hasoc_2020):
    training_data = pd.read_excel(ORIGIN_DATASET_PATH + 'hasoc2020/' + hasoc_2020)
    for index in range(len(training_data)):
        text = training_data['text'][index]
        text = Pre_processing_tweets().clean_unuseful(text)
        training_data['text'][index] = text
    training_data.to_excel(HASOC_2020 + '/' + hasoc_2020, index=False)

#hasoc数据集创建与读取
def create_hasoc2020():
    if os.path.exists(HASOC_2020):
        return
    os.mkdir(HASOC_2020)
    set_hasoc2020_data('hasoc_2020_en_train.xlsx')
    set_hasoc2020_data('hasoc_2020_en_test.xlsx')
    set_hasoc2020_data('hasoc_2020_de_train.xlsx')
    set_hasoc2020_data('hasoc_2020_de_test.xlsx')
    set_hasoc2020_data('hasoc_2020_hi_train.xlsx')
    set_hasoc2020_data('hasoc_2020_hi_test.xlsx')

def get_hasoc2020_data(data_dir):
    text, label = [], []
    data = pd.read_excel(data_dir)
    lens = len(data)
    for index in range(lens):
        item = data.iloc[index]
        if item['text'] != item['text']:
            continue
        text.append(item['text'])
        label.append(1 if item['task1'] == 'HOF' else 0)
    return text, label

def load_hasoc2020():
    en_train_text, en_train_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_en_train.xlsx')
    en_test_text, en_test_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_en_test.xlsx')
    de_train_text, de_train_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_de_train.xlsx')
    de_test_text, de_test_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_de_test.xlsx')
    hi_train_text, hi_train_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_hi_train.xlsx')
    hi_test_text, hi_test_label = get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_hi_test.xlsx')
    return get_hasoc2020_data(HASOC_2020 + '/hasoc_2020_en_train.xlsx'), (en_test_text, en_test_label), (de_train_text, de_train_label), (de_test_text, de_test_label), (hi_train_text, hi_train_label), (hi_test_text, hi_test_label)


"""
EXIST2021数据集相关函数
"""
def set_exist2021_train_data(origin_exist_2021):
    origin_exist_2021 = origin_exist_2021
    training_data = pd.read_csv(origin_exist_2021 + '/train.tsv', sep='\t')
    for index in range(len(training_data)):
        text = training_data.iloc[index]['text']
        text = Pre_processing_tweets().clean_unuseful(text)
        training_data.iloc[index, 4] = text
    training_data.to_csv(TRAIN_EXIST2021, index=False, sep='\t')

def set_exist2021_test_data(origin_exist_2021):
    origin_exist_2021 = origin_exist_2021
    testing_data = pd.read_csv(origin_exist_2021 + '/test.tsv', sep='\t')
    for index in range(len(testing_data)):
        text = testing_data.iloc[index]['text']
        text = Pre_processing_tweets().clean_unuseful(text)
        testing_data.iloc[index, 4] = text
    testing_data.to_csv(TEST_EXIST2021, index=False, sep='\t')

#创建数据集
def create_exist2021():
    if os.path.exists(EXIST2021):
        return
    os.mkdir(EXIST2021)
    origin_exist_2021 = ORIGIN_DATASET_PATH + '/EXIST2021'
    set_exist2021_train_data(origin_exist_2021)
    set_exist2021_test_data(origin_exist_2021)

def get_exist2021_data(type, len):
    type = type
    text, label = [], []
    data_dir = TRAIN_EXIST2021
    if type == 'test':
        data_dir = TEST_EXIST2021
    data =  pd.read_csv(data_dir, sep='\t')
    for index in range(len):
        item = data.iloc[index]
        if item['text'] != item['text']:
            continue
        text.append(item['text'])
        label.append(1 if item['task1'] == 'sexist' else 0)
    return text, label

def get_exist2021_data_temp(type, len):
    type = type
    res = []
    data_dir = TRAIN_EXIST2021
    if type == 'test':
        data_dir = TEST_EXIST2021
    data =  pd.read_csv(data_dir, sep='\t')
    for index in range(len):
        item = data.iloc[index]
        if item['text'] != item['text']:
            continue
        text = item['text']
        label = 1 if item['task1'] == 'sexist' else 0
        res.append((text,label))
    return res

def load_exist2021_data(batch_size, num_steps=280):
    train_data = get_exist2021_data(type='train', len=3437)
    test_data = get_exist2021_data(type='test', len=1000)
    train_tokens = Pre_processing_tweets().tokenize_process(train_data[0])
    test_tokens = Pre_processing_tweets().tokenize_process(test_data[0])
    vocab = Vocab(train_tokens, min_freq=5)
    # print(vocab['girl'])
    train_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens], dtype=torch.int64)
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = load_array(
        (train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = load_array(
        (test_features, torch.tensor(test_data[1])), batch_size, is_train=False)
    return train_iter, test_iter, vocab


#词嵌入模块
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = EMBEDDING_PATH + embedding_name + '.txt'
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(data_dir, 'r', encoding='UTF-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


#加载用于预训练Bert的数据集
