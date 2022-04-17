import time
import torch
from source import evaluate
from tqdm.notebook import tqdm

def train(emb_model, model, loss_fn, optimizer, train_dataloader, val_dataloader=None, epochs=5):
    """Train the CNN model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()
        progress_bar = tqdm(train_dataloader, ascii=True)

        for step, batch in enumerate(progress_bar):

          # Load batch to GPU
          b_input_ids, b_labels = tuple(t.to(device) for t in batch)

          # Get embeddings for current batch
          with torch.no_grad():
            embeddings = emb_model(b_input_ids.to(device))[0]

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

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate.evaluate(emb_model, model, loss_fn, val_dataloader)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")