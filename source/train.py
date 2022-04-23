import time
import torch
from source import evaluate
from tqdm.notebook import tqdm
import numpy as np
import wandb

def train(emb_model, model, loss_fn, optimizer, train_dataloader, val_dataloader=None, epochs=5, bert_layer = 0):
    """Train the CNN model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    results = []


    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0
        train_acc = []

        # Put the model into the training mode
        model.train()
        progress_bar = tqdm(train_dataloader, ascii=True)

        for step, batch in enumerate(progress_bar):
          b_input_ids = batch['input_ids'].to(device)
          b_labels = batch['label'].to(device)
          b_mask = batch['attention_mask'].to(device)

          # Get embeddings for current batch
          with torch.no_grad():
            embeddings = emb_model(b_input_ids, b_mask)[bert_layer]

          # Zero out any previously calculated gradients
          model.zero_grad()

          # Perform a forward pass. This will return logits.
          logits = model(embeddings)

          # Compute loss and accumulate the loss values
          loss = loss_fn(logits, b_labels)
          total_loss += loss.item()

          # Perform a backward pass to calculate gradients
          loss.backward()

          # Update parameters
          optimizer.step()

          # Get the predictions
          preds = torch.argmax(logits, dim=1).flatten()

          # Calculate the accuracy rate
          accuracy = (preds == b_labels).cpu().numpy().mean() * 100
          train_acc.append(accuracy)

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        train_acc = np.mean(train_acc)
        wandb.log({'avg_train_loss':avg_train_loss})
        wandb.log({'train_acc':train_acc})

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate.evaluate(emb_model, model, loss_fn, val_dataloader, bert_layer)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {train_acc:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            wandb.log({'epoch': epoch_i,'val_accuracy':val_accuracy,'val_loss':val_loss})

            results.append([epoch_i, avg_train_loss, train_acc, val_loss, val_accuracy])

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    return np.array(results)
