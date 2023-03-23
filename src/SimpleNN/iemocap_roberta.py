import torch
import numpy as np
from torch import nn
 
from torch.utils.data import DataLoader
import torch

from transformers import BertConfig, AutoTokenizer, AutoModel
import json
import random
import argparse
import torch.nn.functional as F
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# =====================

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class BatchPreprocessor(object): 
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        raw_sentences = []

        # collect all sentences
        for sample in batch:
            raw_sentences.append(sample[0])

        # label processing 
        labels = []
        for sample in batch:
            label = sample[1]
            labels.append(int(label))

        word_ids_from_bert_tokenizer = self.tokenizer(raw_sentences,  padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        return (word_ids_from_bert_tokenizer, torch.LongTensor(labels), raw_sentences) 



class EmotionClassifier(pl.LightningModule):
    def __init__(
        self, 
        pre_trained_model_name='roberta-base', 
        num_labels=1
    ):
        """Initialize."""
        super().__init__() 

        d_model = 768

        # ===================================
        # PUSH YOUR CODE HERE 
        # this is model architecture init process 
    

        # ===================================
        

    def training_step(self, batch, batch_idx, return_y_hat=False):
        model_inputs, labels, raw_sentences = batch

        # ===================================
        # PUSH YOUR CODE HERE 
        # this is forward function
        # y_hat is the output of softmax layer
     

        # ===================================
        

        if return_y_hat:
            return loss, y_hat
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step( batch, batch_idx)
    def test_epoch_end(self, batch_parts):
        return self.validation_epoch_end(batch_parts)
    
    def validation_step(self, batch, batch_idx):
        model_inputs, labels, raw_sentences  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        predictions = torch.argmax(y_hat, dim=1)

        f1_weighted = f1_score(
            labels.cpu(),
            predictions.cpu(),
            average="weighted",
        )
        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels}
    
    def validation_epoch_end(self, batch_parts):
        predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
        labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
        f1_weighted = f1_score(
            labels.cpu(),
            predictions.cpu(),
            average="weighted",
        )
        self.log("valid_f1",f1_weighted, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-5)  

    

#  
# main process 
if __name__ == "__main__":

    # 
    #  init random seed
    set_random_seed(7)
    data_folder= "/home/phuongnm/deeplearning_tutorial/src/SimpleNN/data/"

    # 
    # Label counting
    train_data = json.load(open(f'{data_folder}/iemocap.train.flatten.json'))
    all_labels = []
    for sample in train_data:
        all_labels.append(sample[1])
    # count label 
    num_labels = len(set(all_labels))
    
    # 
    # data loader
    # Load config from pretrained name or path 
    pre_trained_model_name = 'roberta-base'
    bert_tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
    batch_size = 8
    data_loader = BatchPreprocessor(bert_tokenizer)
    test_loader = DataLoader(json.load(open(f"{data_folder}/iemocap.testwindow2.flatten.json")), batch_size=batch_size, collate_fn=data_loader, shuffle=True)
    train_loader = DataLoader(json.load(open(f"{data_folder}/iemocap.trainwindow2.flatten.json")), batch_size=batch_size, collate_fn=data_loader, shuffle=True)
    valid_loader = DataLoader(json.load(open(f"{data_folder}/iemocap.validwindow2.flatten.json")), batch_size=batch_size, collate_fn=data_loader, shuffle=True)
    for e in test_loader:
        print('First epoch data:')
        print('input data\n', e[0])
        print('label data\n',e[1])
        print('padding mask data\n',e[2])
        print(e[0]['input_ids'].device)
        break  
    print('train size', len(train_loader))
    print('test size',  len(test_loader))


    # init trainer 
    trainer = Trainer(max_epochs=3, 
                        accelerator='gpu', 
                        devices=[0], # GPU id, usually is 0 or 1 
                        default_root_dir="./", 
                        val_check_interval=0.1 # 10% epoch, run evaluate one time 
                        )
    
    # init model 
    model = EmotionClassifier(pre_trained_model_name=pre_trained_model_name, num_labels=num_labels)
    
    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # test
    trainer.test(model, test_loader)