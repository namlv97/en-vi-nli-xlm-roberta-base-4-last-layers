import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels,token_type_ids,device):
        self.labels = labels
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.token_type_ids=token_type_ids
        self.device=device
        
    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        token_type_ids = self.token_type_ids[idx]
        sample = {
          "inputs":{
            "input_ids": input_ids.to(self.device),
            'attention_mask':attention_mask.to(self.device),
            'token_type_ids':token_type_ids.to(self.device),
        }, "labels": label.to(self.device)}
        
        return sample

def create_input(tokenizer,token_ids_0,token_ids_1=None,model_type='bert',max_length=512,padding=True):
  num_tokens_0=len(token_ids_0)
  if token_ids_1==None:
    num_tokens_1=0
  else:
    num_tokens_1=len(token_ids_1)

  if model_type=='bert':
    threshold_num_tokens_0=max_length-3-num_tokens_1
  if model_type=='roberta':
    threshold_num_tokens_0=max_length-4-num_tokens_1

  if threshold_num_tokens_0<0:
    return None
  trade_off=abs(threshold_num_tokens_0-num_tokens_0)

  if num_tokens_0>threshold_num_tokens_0:
    
    head_token_ids_0=token_ids_0[:threshold_num_tokens_0]
    tail_token_ids_0=token_ids_0[-threshold_num_tokens_0:]

    head_input_ids=tokenizer.build_inputs_with_special_tokens(head_token_ids_0,token_ids_1)
    head_token_type_ids=tokenizer.create_token_type_ids_from_sequences(head_token_ids_0,token_ids_1)
    head_attention_mask=[1]*len(head_input_ids)

    tail_input_ids=tokenizer.build_inputs_with_special_tokens(tail_token_ids_0,token_ids_1)
    tail_token_type_ids=tokenizer.create_token_type_ids_from_sequences(tail_token_ids_0,token_ids_1)
    tail_attention_mask=[1]*len(tail_input_ids)
    
    input_ids=[head_input_ids,tail_input_ids]
    token_type_ids=[head_token_type_ids,tail_token_type_ids]
    attention_mask=[head_attention_mask,tail_attention_mask]
  else:
    input_ids=tokenizer.build_inputs_with_special_tokens(token_ids_0,token_ids_1)
    token_type_ids=tokenizer.create_token_type_ids_from_sequences(token_ids_0,token_ids_1)
    attention_mask=[1]*len(input_ids)

    if padding==True:
      input_ids=input_ids+[tokenizer.pad_token_id]*trade_off
      attention_mask=attention_mask+[0]*trade_off
      token_type_ids=token_type_ids+[0]*trade_off

    input_ids=[input_ids]
    attention_mask=[attention_mask]
    token_type_ids=[token_type_ids]

  return input_ids,attention_mask,token_type_ids


def create_inputs(tokenizer,list_sentences_0,list_sentences_1,list_labels,model_type='bert',max_length=512):
  
  input_ids=[]
  labels=[]
  bar=tqdm(list_sentences_0)
  attention_mask=[]
  token_type_ids=[]
  num_training_samples=len(list_sentences_0)
  for i in range(num_training_samples):
    token_ids_0=tokenizer.encode(list_sentences_0[i],add_special_tokens=False)
    if list_sentences_1==None:
      token_ids_1=None
    else:
      token_ids_1=tokenizer.encode(list_sentences_1[i],add_special_tokens=False)
    _input_ids,_attention_mask,_token_type_ids=create_input(tokenizer,token_ids_0,token_ids_1,model_type,max_length)
    
    _labels=[list_labels[i]]*len(_input_ids)
    labels+=_labels
    input_ids+=_input_ids
    attention_mask+=_attention_mask
    token_type_ids+=_token_type_ids
    bar.update()  
  return {
    'input_ids':torch.Tensor(input_ids).to(torch.long),
    'attention_mask':torch.Tensor(attention_mask).to(torch.int),
    'token_type_ids':torch.Tensor(token_type_ids).to(torch.int),
    'labels':torch.Tensor(labels).to(torch.long)}

def create_dataloader(tokenizer,list_sentences_0,list_sentences_1,list_labels,model_type='bert',max_length=512,batch_size=32,drop_last=False,shuffle=True,device='cpu'):
  dataloader=create_inputs(tokenizer,list_sentences_0,list_sentences_1,list_labels,model_type,max_length)
  
  dataloader=ClassificationDataset(**dataloader,device=device)
  dataloader=DataLoader(dataloader,batch_size=batch_size,drop_last=drop_last,shuffle=shuffle)
  return dataloader