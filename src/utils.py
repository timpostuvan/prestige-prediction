import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def train(model, loss_fcn, device, optimizer, num_epochs, train_dataloader, val_dataloader):
    epoch_list = []
    scores_list = []
    best_val_score = float("inf")
    best_model = model

    # loop over epochs
    for epoch in range(num_epochs):
        model.train()
        losses = []
        # loop over batches
        for i, train_batch in enumerate(train_dataloader):
            # if all nodes are masked out, skip the batch
            mask = train_batch.mask
            num = torch.sum((mask == True).float())
            if num == 0:
                continue

            optimizer.zero_grad()
            train_batch_device = train_batch.to(device)

            # logits is the output of the model
            logits = model(train_batch_device.x, train_batch_device.edge_index)

            # compute the loss
            loss = loss_fcn(logits[mask].flatten(), train_batch_device.y[mask])
            
            # optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            # evaluate the model on the validation set
            # computes the f1-score (see next function)
            score = evaluate(model, loss_fcn, device, val_dataloader)
            print("MSE: {:.4f}".format(score))
            if best_val_score > score:
                best_val_score = score
                best_model = copy.deepcopy(model)

            scores_list.append(score)
            epoch_list.append(epoch)

    return epoch_list, scores_list, best_model

def evaluate(model, loss_fcn, device, dataloader):
    all_predictions, all_labels = [], []

    model.eval()
    for i, batch in enumerate(dataloader):
        # if all nodes are masked out, skip the batch
        mask = batch.mask
        num = torch.sum((mask == True).float())
        if num == 0:
            continue

        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        loss_test = loss_fcn(output[mask].flatten(), batch.y[mask])
        
        all_predictions.append(output[mask].detach().cpu().numpy())
        all_labels.append(batch.y[mask].detach().cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    return mean_squared_error(all_labels, all_predictions)

def plot_MSE_scores(epoch_lists: list, MSE_scores: list, model_name: str):
    plt.figure(figsize = [10,5])
    for i in range(len(epoch_lists)):
        plt.plot(epoch_lists[i], MSE_scores[i], 'b', label=f"{model_name}-{i}")

    plt.title("Evolution of MSE score w.r.t epochs")
    plt.ylim([0.0, 1.0])
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()