import torch
import numpy as np
from tqdm.notebook import tqdm
import wandb

def evaluate(emb_model, model, loss_fn, val_dataloader, bert_layer = 0, epoch_i = 0):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    progress_bar = tqdm(val_dataloader, ascii=True)
    for idx, batch in enumerate(progress_bar):
        # Load batch to GPU
        b_input_ids = batch['input_ids'].to(device)
        b_labels = batch['label'].to(device)
        b_mask = batch['attention_mask'].to(device)

        # init empty np array for labels/preds for epoch
        if (idx == 0):
          labels = np.array([])
          predictions = np.array([])

        # Get embeddings for current batch
        with torch.no_grad():
            embeddings = emb_model(b_input_ids, b_mask)[bert_layer]

        # Compute logits
        with torch.no_grad():
            logits = model(embeddings)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        labels = np.concatenate((labels, b_labels.cpu().numpy()), axis=0)
        predictions = np.concatenate((predictions, preds.cpu().numpy()), axis=0)
        val_accuracy.append(accuracy)
        #wandb.log({"pr": wandb.plot.pr_curve(np_labels, np_preds )})
        #wandb.log({"roc": wandb.plot.roc_curve(np_labels, np_preds )})

    # Compute the average accuracy and loss over the validation set.
    # Get a confusion matrix for the epoch
    cm = wandb.plot.confusion_matrix( y_true=labels, preds=predictions ,class_names=['Real','Fake'])
    wandb.log({"conf_mat_epoch" + str(epoch_i): cm})
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy
