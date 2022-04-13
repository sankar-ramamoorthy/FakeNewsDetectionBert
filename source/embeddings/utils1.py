
#Code: load pretrained embeddings (BERT)
from time import time
import torch


def tokenize_text(text_arr, max_seq):
    return [tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in text_arr.values]

def pad_text(tokenized_text, max_seq):
    return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])

def tokenize_and_pad_text(text_arr, max_seq):
    tokenized_text = tokenize_text(text_arr, max_seq)
    padded_text = pad_text(tokenized_text, max_seq)
    return torch.tensor(padded_text)

def targets_to_tensor(label_arr):
    return torch.tensor(label_arr.values, dtype=torch.long)


def load_pretrained_embeddings(tests,labels,max_seq_len=200,tran_fp='train.csv',test_fp='test.csv',pretrained_fp='bert-base-uncased'):
    # inputs
    # train_fp = 'train.csv'
    # test_fp = 'test.csv'
    # pretrained_fp = 'bert-base-uncased'
    # 
    # 
    # max_seq_len = 200
    # texts = samp.text
    # labels = samp.label
        

    tokenizer = BertTokenizer.from_pretrained(pretrained_fp)
    bert_model = BertModel.from_pretrained(pretrained_fp)
    # enc = tokenizer.encode(X_train.values[0], add_special_tokens=True)
    enc = tokenizer.encode(texts.values[0], add_special_tokens=True)



        # warning comes up because sequences are longer,
    # but this function also clips them to max_seq_len,
    # so it won't be a problem in the model
    input_idxs = tokenize_and_pad_text(texts, max_seq_len)



    # get contextualized embeddings from bert model

    start = time()
    with torch.no_grad():
        bert_embeddings = bert_model(input_idxs)[0]
        # X_train_bert = bert_model(train_indices)[0]  # Models outputs are tuples
        # X_val_bert = bert_model(val_indices)[0]
        # X_test_bert = bert_model(test_indices)[0]
    end = time()
    elapsed = end - start
    if elapsed < 180:
        print(f'code took {elapsed:0.2f} seconds to execute')
    else:
        print(f'code took {elapsed / 60:0.2f} minutes to execute')

    bert_labels = targets_to_tensor(labels)
    # y_train_bert = targets_to_tensor(y_train)
    # y_val_bert = targets_to_tensor(y_val)
    # y_test_bert = targets_to_tensor(y_test)
    return(bert_embeddings,bert_labels)
