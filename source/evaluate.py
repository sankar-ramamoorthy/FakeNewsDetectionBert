import torch
import numpy as np
from tqdm.notebook import tqdm

def evaluate(emb_model, model, loss_fn, val_dataloader):
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
    for batch in progress_bar:
        # Load batch to GPU
        b_input_ids = batch['input_ids'].to(device)
        b_labels = batch['label'].to(device)
        b_mask = batch['attention_mask'].to(device)

        # Get embeddings for current batch
        with torch.no_grad():
            embeddings = emb_model(b_input_ids, b_mask)[0]

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
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy