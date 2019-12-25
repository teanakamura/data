from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import numpy as np
import os
from tqdm import tqdm

from IPython import embed

class MyDataset(Dataset):
  def __init__(self, data):
    self.data = data
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    return self.data[index]

def collate_fn(batch):
  return pad_sequence(batch, batch_first=True).to(device)


home = os.environ['HOME']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = 'cpu'
model = torch.load(home + '/BERT-Keyword-Extractor/model.pt')
model.eval()
model.to(device)

data_num = []
sent = []
with open(home+'/Data/cnndm-pj/raw_summary/test.sum') as f:
  for l in f:
    n, s = l.split('\t', 1)
    data_num.append(n)
    sent.append(s)
tsent = [tokenizer.tokenize(s) for s in tqdm(sent)]
#print(tsent)
dataset = MyDataset([torch.tensor(tokenizer.convert_tokens_to_ids(s)) for s in tsent])
bs = 2
dl = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn)

#sampler = BatchSampler(SequentialSampler(tsent), batch_size=3, drop_last=False)

for b, batch in enumerate(dl):
#isent = pad_sequence([torch.tensor(tokenizer.convert_tokens_to_ids(s)) for s in batch], batch_first=True).to(device)
  attention_mask = batch!=0
  #print(batch)
  #print(attention_mask)
  inp = batch
  outp = model(inp, attention_mask=attention_mask)
  #print(outp)
  outp = outp.detach().to('cpu')
  pred = np.argmax(outp, axis=2)
  #print(pred)
  pred_wh = np.where(pred < 2) 
  for i, j in zip(*pred_wh):
    sent_num = b*bs+i
    n = data_num[sent_num]
    print(n, j, tsent[sent_num][j])
