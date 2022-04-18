from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import tokenizers
import torch
import os



'''
This class will generate a HuggingFace dataset from a data folder containing two files:  train.csv and test.csv
Need to perform additional tests if using a validation file. 

The data_folder default parameter parameter assumes that the Jupyter Notebook is running at the Project level of the Github code. 
(ie. '/content/drive/MyDrive/CS7643/Project' in a local google drive containing the code)

The class contains a method ('get_data_loader') to return a data_loader for train/validation 

'''
class bert_data:
  def __init__(self, data_folder = 'data', train_file = 'train.csv', test_file = 'test.csv', val_file = None, tokenizer_type = 'bert-base-uncased', data_tokens = None):

    #print(os.getcwd())
    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_type)

    if val_file is not None:
      self.dataset = load_dataset(data_folder, data_files = {'train' : train_file, 'test': test_file, 'valid': val_file })
    else:
      self.dataset = load_dataset(data_folder, data_files = {'train' : train_file, 'test': test_file})

    # At the moment, first column in the dataset is unnamed, rename as id (delete if no longer needed)
    self.dataset = self.dataset.rename_column("Unnamed: 0", 'id')

    self.data_loader = None

    # If data_tokens are not passed in, this will be None
    self.data_tokens = data_tokens



  '''
  Tokenize the class parameter self.dataset using the initialized tokenizer and return the tokenized result.
  
  Train/Test/Validation do not need to be specified as they will all be tokenized in one go with this method.

  Dataset is returned as below (assuming no validation in dataset):
      {'test': ['id',
      'text',
      'label',
      'input_ids',
      'token_type_ids',
      'attention_mask'],
    'train': ['id',
      'text',
      'label',
      'input_ids',
      'token_type_ids',
      'attention_mask']}

  '''
  def tokenize(self, col_tokenize = 'text', add_special_tokens = True, max_length = 64, truncation = True, padding = 'max_length'):

    # If data model was initiated with tokens already, no need to recalculate
    if self.data_tokens is not None:
      print('skipping tokenize step and loading saved tokens...')
      return self.data_tokens

    self.data_tokens = self.dataset.map(lambda e: self.tokenizer(e[col_tokenize], \
                    add_special_tokens = add_special_tokens, \
                    max_length = max_length, \
                    truncation = truncation, \
                    padding = padding))
    
    # Make dataset torch format for bert model params
    self.data_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    return self.data_tokens

  '''
  Get train and validation data_loaders. 
  Expects to receive the train['input_ids'], validation['input_ids], train['label'], validation['label'] as Tensor parameters. 
  The train/validation inputs will be tensors if generated from the above tokenizer method. 

  '''
  def get_data_loader(self, train_inputs, val_inputs, train_labels, val_labels,
                batch_size=64):
    """Load train and validation sets to DataLoader.
    """

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader

  '''
  Returns initiated tokenizer
  '''
  def get_tokenizer(self):
    return self.tokenizer

  '''
  Returns dataset after tokenization
  '''
  def get_tokens(self):
    return self.data_token

  '''
  Returns dataset before tokenization
  '''
  def get_dataset(self):
    return self.dataset




