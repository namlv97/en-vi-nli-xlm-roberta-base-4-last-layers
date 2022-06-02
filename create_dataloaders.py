from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_mask,labels,device,token_type_ids=None):
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
        sample = {"inputs":{"input_ids": input_ids.to(self.device),'attention_mask':attention_mask.to(self.device)}, "labels": label.to(self.device)}
        if self.token_type_ids!=None:
          sample['inputs'].update({'token_type_ids':self.token_type_ids[idx].to(self.device)})
        return sample

class SiameseClassificationDataset(Dataset):
    def __init__(self, input_ids_1, attention_mask_1,input_ids_2, attention_mask_2,labels,device,token_type_ids_1=None,token_type_ids_2=None):
        self.labels = labels

        self.input_ids_1=input_ids_1
        self.attention_mask_1=attention_mask_1
        self.token_type_ids_1=token_type_ids_1

        self.input_ids_2=input_ids_2
        self.attention_mask_2=attention_mask_2
        self.token_type_ids_2=token_type_ids_2

        self.device=device
        
    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_ids_1 = self.input_ids_1[idx]
        attention_mask_1 = self.attention_mask_1[idx]
        input_ids_2 = self.input_ids_2[idx]
        attention_mask_2 = self.attention_mask_2[idx]
        sample = {"inputs":{"input_ids_1": input_ids_1.to(self.device),'attention_mask_1':attention_mask_1.to(self.device),
                            "input_ids_2": input_ids_2.to(self.device),'attention_mask_2':attention_mask_2.to(self.device)},
                   "labels": label.to(self.device)}
        if self.token_type_ids_1!=None:
          sample['inputs'].update({'token_type_ids_1':self.token_type_ids_1[idx].to(self.device)})
        if self.token_type_ids_2!=None:
          sample['inputs'].update({'token_type_ids_2':self.token_type_ids_2[idx].to(self.device)})
        return sample

class EnsembleClassificationDataset(Dataset):
    def __init__(self, input_ids_1, attention_mask_1,input_ids_2, attention_mask_2,labels,device,token_type_ids_1=None,token_type_ids_2=None):
        self.labels = labels

        self.input_ids_1=input_ids_1
        self.attention_mask_1=attention_mask_1
        self.token_type_ids_1=token_type_ids_1

        self.input_ids_2=input_ids_2
        self.attention_mask_2=attention_mask_2
        self.token_type_ids_2=token_type_ids_2

        self.device=device
        
    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_ids_1 = self.input_ids_1[idx]
        attention_mask_1 = self.attention_mask_1[idx]
        input_ids_2 = self.input_ids_2[idx]
        attention_mask_2 = self.attention_mask_2[idx]
        sample = {"inputs":{"input_ids_1": input_ids_1.to(self.device),'attention_mask_1':attention_mask_1.to(self.device),
                            "input_ids_2": input_ids_2.to(self.device),'attention_mask_2':attention_mask_2.to(self.device)},
                   "labels": label.to(self.device)}
        if self.token_type_ids_1!=None:
          sample['inputs'].update({'token_type_ids_1':self.token_type_ids_1[idx].to(self.device)})
        if self.token_type_ids_2!=None:
          sample['inputs'].update({'token_type_ids_2':self.token_type_ids_2[idx].to(self.device)})
        return sample

def create_input(tokenizer,token_ids_0,token_ids_1=None,max_length=512,padding=True):
  num_tokens_0=len(token_ids_0)
  num_tokens_1=len(token_ids_1)

  if tokenizer.name_or_path in ['xlm-roberta-base']:
    threshold_num_tokens_0=max_length-4-num_tokens_1
  else:
    threshold_num_tokens_0=max_length-3-num_tokens_1


  token_type_ids=[]

  if threshold_num_tokens_0<0:
    return None
  trade_off=abs(threshold_num_tokens_0-num_tokens_0)
  if num_tokens_0>threshold_num_tokens_0:
    
    head_token_ids_0=token_ids_0[:threshold_num_tokens_0]
    tail_token_ids_0=token_ids_0[-threshold_num_tokens_0:]

    head_input_ids=tokenizer.build_inputs_with_special_tokens(head_token_ids_0,token_ids_1)
    if tokenizer.name_or_path not in ['xlm-roberta-base']:
      head_token_type_ids=tokenizer.create_token_type_ids_from_sequences(head_token_ids_0,token_ids_1)
      token_type_ids.append(head_token_type_ids)
    head_attention_mask=[1]*len(head_input_ids)

    tail_input_ids=tokenizer.build_inputs_with_special_tokens(tail_token_ids_0,token_ids_1)
    if tokenizer.name_or_path not in ['xlm-roberta-base']:
      tail_token_type_ids=tokenizer.create_token_type_ids_from_sequences(tail_token_ids_0,token_ids_1)
      token_type_ids.append(tail_token_type_ids)
    tail_attention_mask=[1]*len(tail_input_ids)
    
    input_ids=[head_input_ids,tail_input_ids]

    attention_mask=[head_attention_mask,tail_attention_mask]
  else:
    input_ids=tokenizer.build_inputs_with_special_tokens(token_ids_0,token_ids_1)
    if tokenizer.name_or_path not in ['xlm-roberta-base']:
      token_type_ids=tokenizer.create_token_type_ids_from_sequences(token_ids_0,token_ids_1)
    attention_mask=[1]*len(input_ids)
    if padding==True:
      input_ids=input_ids+[1]*trade_off
      attention_mask=attention_mask+[0]*trade_off
      token_type_ids=token_type_ids+[0]*trade_off
    input_ids=[input_ids]
    attention_mask=[attention_mask]
    if tokenizer.name_or_path not in ['xlm-roberta-base']:
      token_type_ids=[token_type_ids]
  return input_ids,attention_mask,token_type_ids

def create_inputs(tokenizer,list_sentences_0,list_sentences_1,list_labels,max_length=512):
  
  input_ids=[]
  labels=[]
  bar=tqdm(list_sentences_0)
  attention_mask=[]
  token_type_ids=[]
  num_training_samples=len(list_sentences_0)
  for i in range(num_training_samples):
    token_ids_0=tokenizer.encode(list_sentences_0[i],add_special_tokens=False)
    token_ids_1=tokenizer.encode(list_sentences_1[i],add_special_tokens=False)
    
    _input_ids,_attention_mask,_token_type_ids=create_input(tokenizer,token_ids_0,token_ids_1,max_length)
    
    _labels=[list_labels[i]]*len(_input_ids)
    labels+=_labels
    input_ids+=_input_ids
    attention_mask+=_attention_mask
    token_type_ids+=_token_type_ids
    bar.update()  

  outputs={
      'input_ids':torch.Tensor(input_ids).to(torch.long),
      'attention_mask':torch.Tensor(attention_mask).to(torch.int),
      'labels':torch.Tensor(labels).to(torch.long)
  }
  if tokenizer.name_or_path!='xlm-roberta-base':
    outputs.update({'token_type_ids':torch.Tensor(token_type_ids).to(torch.int)})
  return outputs

def create_dataloader(tokenizer,list_sentences_0,list_sentences_1,list_labels,max_length=512,batch_size=32,drop_last=False,shuffle=True,device='cpu'):
  dataloader=create_inputs(tokenizer,list_sentences_0,list_sentences_1,list_labels,max_length)

  dataloader=ClassificationDataset(**dataloader,device=device)
  dataloader=DataLoader(dataloader,batch_size=batch_size,drop_last=drop_last,shuffle=shuffle)
  return dataloader

def create_siamese_dataloader(tokenizer,list_sentences_0,list_sentences_1,list_labels,max_length=512,batch_size=32,drop_last=False,shuffle=True,device='cpu'):
  dataloader_1=tokenizer(list_sentences_0,return_tensors='pt',max_length=max_length,padding=True,truncation=True)
  dataloader_2=tokenizer(list_sentences_0,return_tensors='pt',max_length=max_length,padding=True,truncation=True)
  dataloader={
      'input_ids_1':dataloader_1['input_ids'],
      'attention_mask_1':dataloader_1['attention_mask'],
      'input_ids_2':dataloader_2['input_ids'],
      'attention_mask_2':dataloader_2['attention_mask'],
      'labels':torch.Tensor(list_labels).to(torch.long)
  }
  dataloader=SiameseClassificationDataset(**dataloader,device=device)
  dataloader=DataLoader(dataloader,batch_size=batch_size,drop_last=drop_last,shuffle=shuffle)
  return dataloader

def create_enssemble_dataloader(list_tokenizers,list_sentences_0,list_sentences_1,list_labels,max_length=512,batch_size=32,drop_last=False,shuffle=True,device='cpu'):
  
  dataloaders=[create_inputs(tokenizer,list_sentences_0,list_sentences_1,list_labels,max_length)for tokenizer in list_tokenizers]
  
  dataloader={}
  for idx,d in enumerate(dataloaders):
    for k,v in d:
      if k!='labels':
        dataloader[f'{k}_{idx+1}']=v

  dataloader['labels']=torch.Tensor(list_labels).to(torch.long)
      
  dataloader=EnsembleClassificationDataset(**dataloader,device=device)
  dataloader=DataLoader(dataloader,batch_size=batch_size,drop_last=drop_last,shuffle=shuffle)
  return dataloader