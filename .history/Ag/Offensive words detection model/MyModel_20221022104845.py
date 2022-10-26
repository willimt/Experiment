
import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=True)

def preprocessing_for_bert(data):
    input_ids, attention_masks = [], []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=60,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.Tensor(input_ids)
    attention_masks = torch.Tensor(attention_masks)
    # print(attention_masks)
    return input_ids, attention_masks

class MyBertModel(nn.Module):

    def __init__(self, len, class_size, pretrained_name='bert-base-uncased'):
        super(MyBertModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_name)
        config.output_hidden_states = True
        # config.output_attentions = True
        config.return_dict = True
        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        self.w_h = nn.Parameter(torch.Tensor(len))
        self.b_n = nn.Parameter(torch.Tensor(768))
        self.W_h = nn.Parameter(torch.Tensor(768, 768))
        self.W_e = nn.Parameter(torch.Tensor(768, 768))
        self.W_hs = nn.Parameter(torch.Tensor(768, 768))
        self.W_L1 = nn.Parameter(torch.Tensor(768 * 3, 768))
        self.b_L1 = nn.Parameter(torch.Tensor(len, 768))
        self.W_L2 = nn.Parameter(torch.Tensor(768, class_size))
        self.b_L2 = nn.Parameter(torch.Tensor(len, class_size))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.len = len
        self.softmax = nn.Softmax(dim=-1)
        self.classifer = nn.Linear(len, class_size)
        nn.init.uniform_(self.w_h, -0.1, 0.1)
        nn.init.uniform_(self.W_h, -0.1, 0.1)
        nn.init.uniform_(self.W_e, -0.1, 0.1)
        nn.init.uniform_(self.b_n, -0.1, 0.1)
        nn.init.uniform_(self.W_hs, -0.1, 0.1)
        nn.init.uniform_(self.W_L1, -0.1, 0.1)
        nn.init.uniform_(self.b_L1, -0.1, 0.1)
        nn.init.uniform_(self.W_L2, -0.1, 0.1)
        nn.init.uniform_(self.b_L2, -0.1, 0.1)

    def forward(self, inputs):
        input_ids, 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state_cls = outputs[0]  # Last_hidden size [32, 80, 768]
        Embedding = outputs[2][0]  # Embedding size [32, 80, 768]
        hs = self.tanh(torch.matmul(Embedding.permute(0, 2, 1),
                       self.w_h) + self.b_n)  # hs size [32, 768]
        hs = torch.repeat_interleave(torch.unsqueeze(
            hs, 1), self.len, 1)  # hs size [32, 80, 768]
        h_f = torch.cat((torch.cat((torch.matmul(last_hidden_state_cls, self.W_h), torch.matmul(
            Embedding, self.W_e)), dim=-1), torch.matmul(hs, self.W_hs)), dim=-1)
        # hf size [32, 80, 768 * 3]
        L1 = self.relu(torch.matmul(h_f, self.W_L1) +
                       self.b_L1)  # L1 size [32, 80, 768]
        L2 = self.softmax(torch.matmul(L1, self.W_L2) +
                          self.b_L2)  # L2 size [32, 80, 2]
        outs = L2[:, :, 1]
        return outs
