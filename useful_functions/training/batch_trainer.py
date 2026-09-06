import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, criterion, metrics, epochs=10, batch_size=64):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X, y):
        # Removed .numpy() calls as X and y are already numpy arrays
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ensure labels are long for CrossEntropyLoss
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size)
        
        losses_train, losses_test = [], []
        accuracies_train, accuracies_test = [], []
    
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.run_epoch(train_loader, mode="train")
            test_loss, test_acc = self.run_epoch(test_loader, mode="test")
    
            losses_train.append(train_loss)
            losses_test.append(test_loss)
            accuracies_train.append(train_acc)
            accuracies_test.append(test_acc)
    
            print(f"Epoch [{epoch}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")
    
        self.plot_metrics(losses_train, losses_test, "Loss Curves", "Loss")
        self.plot_metrics(accuracies_train, accuracies_test, "Accuracy Curves", "Accuracy")
    
    def run_epoch(self, dataloader: DataLoader, mode: str):
        is_train = (mode == "train")
        self.model.train() if is_train else self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.set_grad_enabled(is_train):
            for batch_X, batch_y in dataloader:
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
        
        return total_loss / total_samples, correct / total_samples

    def plot_metrics(self, train_vals, test_vals, title: str, ylabel: str):
        plt.figure(figsize=(8, 5))
        epochs_range = np.arange(1, len(train_vals) + 1)
        plt.plot(epochs_range, train_vals, label="Train", marker='o')
        plt.plot(epochs_range, test_vals, label="Test", marker='s')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
