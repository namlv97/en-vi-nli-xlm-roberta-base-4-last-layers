import torch
from torch import nn
from transformers import RemBertModel,RemBertPreTrainedModel,RobertaModel,RobertaPreTrainedModel

class RemBert4LastLayersForClassification(RemBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.hidden_size=config.hidden_size
        self.rembert = RemBertModel(config, add_pooling_layer=False)

        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)
        
        # Initialize weights and apply final processing
        self.init_weights()

    
    def forward(self,input_ids,attention_mask,token_type_ids,labels=None):

        outputs = self.rembert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        last_layers=torch.stack(outputs.hidden_states[-4:])
        last_layers=last_layers.permute(1,2,0,3)
        features=torch.flatten(last_layers,start_dim=2)
        cls_embedding=features[:,0,:]
        
        logits = self.classifier(cls_embedding)

        return logits

class XLMRoBERTa4LastLayersForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.hidden_size=config.hidden_size
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.classifier = nn.Linear(self.config.hidden_size*4, self.config.num_labels)
        
        # Initialize weights and apply final processing
        self.init_weights()

    
    def forward(self,input_ids,attention_mask,token_type_ids,labels=None):

        outputs = self.roberta(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        last_layers=torch.stack(outputs.hidden_states[-4:])
        last_layers=last_layers.permute(1,2,0,3)
        features=torch.flatten(last_layers,start_dim=2)
        cls_embedding=features[:,0,:]
        
        logits = self.classifier(cls_embedding)
        return logits