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
        model_configs
    ):
        """Initialize."""
        super().__init__() 

        # ===================================
        # PUSH YOUR CODE HERE 
        # this is model architecture init process 
        self.save_hyperparameters(model_configs)
        self.model_configs = model_configs
    
        self.model = AutoModel.from_pretrained(model_configs.pre_trained_model_name)
        d_model = self.model.config.hidden_size
        
        self.dropout_layer = nn.Dropout(model_configs.dropout)

        self.output_layer = nn.Linear(d_model, model_configs.num_labels)
        self.softmax_layer = nn.Softmax(dim=1)
        self.loss_computation = torch.nn.CrossEntropyLoss()

        # ===================================
        

    def training_step(self, batch, batch_idx, return_y_hat=False):
        model_inputs, labels, raw_sentences = batch

        # ===================================
        # PUSH YOUR CODE HERE 
        # this is forward function
        # y_hat is the output of softmax layer
     

        model_inputs, labels, raw_sentences = batch
        outputs = self.model(**model_inputs)
    
        h_cls = self.dropout_layer(outputs[1])
        y_hat = self.softmax_layer(self.output_layer(h_cls))

        loss = self.loss_computation(y_hat, labels)
        # ===================================
        self.log('train_loss', loss)
        

        if return_y_hat:
            return loss, y_hat
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step( batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        model_inputs, labels, raw_sentences  = batch
        loss, y_hat = self.training_step(batch, batch_idx, return_y_hat=True)
        predictions = torch.argmax(y_hat, dim=1)

        return {'val_loss_step': loss, 'y_hat': y_hat, 'labels': labels}
    
    def _eval_epoch_end(self, batch_parts):
        predictions = torch.cat([torch.argmax(batch_output['y_hat'], dim=1) for batch_output in batch_parts],  dim=0)
        labels = torch.cat([batch_output['labels']  for batch_output in batch_parts],  dim=0)
        f1_weighted = f1_score(
            labels.cpu(),
            predictions.cpu(),
            average="weighted",
        )
        return f1_weighted*100
    
    def validation_epoch_end(self, batch_parts):
        self.log('valid_f1', self._eval_epoch_end(batch_parts), prog_bar=True)
    def test_epoch_end(self, batch_parts):
        self.log('test_f1', self._eval_epoch_end(batch_parts), prog_bar=True)

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr = 1e-5, weight_decay=0.001)  
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                            betas=(0.9, 0.98),  # according to RoBERTa paper
                            lr=self.model_configs.lr,
                        eps=1e-06, weight_decay=0.001)
        
        num_gpus = 1
        max_ep=self.model_configs.max_epochs
        t_total = (len(train_loader) // (1 * num_gpus) + 1) * max_ep
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.model_configs.lr, pct_start=float(0/t_total),
            final_div_factor=10000,
            total_steps=t_total, anneal_strategy='linear'
        ) 
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    

#  
# main process 
if __name__ == "__main__":

    # 
    #  init random seed
    set_random_seed(7)
    data_folder= "/home/phuongnm/deeplearning_tutorial/src/SimpleNN/data/raw_data/"

    # 
    # Label counting
    data_name_pattern = "iemocap.{}.contextLimit512.json"
    train_data = json.load(open(f'{data_folder}/{data_name_pattern.format("train")}'))
    all_labels = []
    for sample in train_data:
        all_labels.append(sample[1])
    # count label 
    num_labels = len(set(all_labels))
    
    # 
    # data loader
    # Load config from pretrained name or path 
    model_configs = argparse.Namespace(
        pre_trained_model_name = 'roberta-large',
        batch_size = 8,
        max_epochs=5,
        dropout=0.2,
        lr = 1e-5,
        num_labels=num_labels,
        data_name_pattern = data_name_pattern
    )
    
    bert_tokenizer = AutoTokenizer.from_pretrained(model_configs.pre_trained_model_name)
    data_loader = BatchPreprocessor(bert_tokenizer)
    test_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('test')}")), batch_size=model_configs.batch_size, collate_fn=data_loader, shuffle=True)
    train_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('train')}")), batch_size=model_configs.batch_size, collate_fn=data_loader, shuffle=True)
    valid_loader = DataLoader(json.load(open(f"{data_folder}/{data_name_pattern.format('valid')}")), batch_size=model_configs.batch_size, collate_fn=data_loader, shuffle=True)
    for e in test_loader:
        print('First epoch data:')
        print('input data\n', e[0])
        print('label data\n',e[1])
        print('padding mask data\n',e[2])
        print(e[0]['input_ids'].device)
        break  
    print('train size', len(train_loader))
    print('test size',  len(test_loader))

    checkpoint_callback = ModelCheckpoint(dirpath="./", save_top_k=1, 
                                        auto_insert_metric_name=True, 
                                        mode="max", 
                                        monitor="valid_f1", 
                                        filename=model_configs.pre_trained_model_name+"-iemocap-nodrop-{valid_f1:.2f}",
                                    #   every_n_train_steps=opts.ckpt_steps
                                        )
    

    # init trainer 
    trainer = Trainer(max_epochs=model_configs.max_epochs, 
                        accelerator='gpu', 
                        devices=[0], # GPU id, usually is 0 or 1
                        callbacks=[checkpoint_callback],
                        default_root_dir="./", 
                        val_check_interval=0.1 # 10% epoch, run evaluate one time 
                        )
    
    # init model 
    model = EmotionClassifier(model_configs)
    
    # train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # test
    trainer.test(model, test_loader, ckpt_path=checkpoint_callback.best_model_path)
    # trainer.test(model, test_loader, ckpt_path='/home/phuongnm/deeplearning_tutorial/roberta-large-iemocap-valid_f1=61.44.ckpt')
