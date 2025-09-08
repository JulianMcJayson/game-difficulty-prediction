from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import joblib

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

class GameResultDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y[index], dtype=torch.float32).to(DEVICE)
        return x, y

class GamePredictionModel(nn.Module):
    def __init__(self, input_dim : int = 2, hidden_dim: int = 2048, output_dim : int = 1, dropout_rate : float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.PReLU(),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//8, hidden_dim//16),
            nn.PReLU(),
            nn.Linear(hidden_dim//16, hidden_dim//32),
            nn.PReLU(),
            nn.Linear(hidden_dim//32, output_dim)
        )
    def forward(self, x):
        outputs = self.net(x)
        return outputs

def train_and_test(dataset, epoch : int):
    X = [[np.log1p(t["fail"]), t["activity"]] for t in dataset]
    y = [[float(t["adaptive_difficulty"])] for t in dataset]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    x_scalar = StandardScaler()
    x_train_scale= x_scalar.fit_transform(x_train)
    x_test_scale = x_scalar.transform(x_test)
    train_data, test_data = GameResultDataset(x_train_scale, y_train), GameResultDataset(x_test_scale, y_test)
    train_dataset, test_dataset = DataLoader(train_data, batch_size=2048, shuffle=True), DataLoader(test_data, batch_size=2048, shuffle=True)
    model = GamePredictionModel().to(device=DEVICE).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=50)
    loss_fn = torch.nn.BCEWithLogitsLoss().to(DEVICE)

    train_losses = []
    val_losses = []
    val_accuracy = []

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 100

    for e in range(epoch):
        model.train()
        train_loss = 0.00
        for _ , (x, y) in enumerate(train_dataset):
            # y = y.squeeze()
            optim.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.00
        correct_prediction = 0.0
        total_prediction = 0.0

        with torch.no_grad():
            for x, y in test_dataset:
                # y = y.squeeze()
                output = model(x)
                loss = loss_fn(output, y)
                val_loss += loss.item()
                # _, predict_table = torch.max(output.data, 1)
                predict_table = (torch.sigmoid(output) > 0.5).float()
                correct_prediction += (predict_table == y).sum().item()
                total_prediction += y.size(0)
        
        avg_val_loss = val_loss / len(test_dataset)
        val_losses.append(avg_val_loss)
        accuracy = correct_prediction / total_prediction
        val_accuracy.append(accuracy)

        schedule.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), "model/best_model.pth")
        else:
            patience_counter += 1
        
        if patience_counter > early_stop_patience:
            print(f"Early stopping at epoch {e+1}")
            break 

        current_lr = optim.param_groups[0]['lr']
        print(f'Epoch {e+1}/{epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}')

    print("Training complete. Evaluating on test set...")

    model.eval()

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for _, (x,y) in enumerate(test_dataset):
            y = y.squeeze()
            output = model(x)
            # predict_table = torch.argmax(output, 1)
            predict_table = (torch.sigmoid(output) > 0.5).float()
            all_predictions.extend(predict_table.cpu().numpy())
            all_true_values.extend(y.cpu().numpy())

    print(f'Final Test Accuracy: {accuracy_score(all_true_values, all_predictions):.4f}\n')
    print("Classification Report:")

    target_names = ["0 (Decrease)", "1 (Increase)"]
    print(classification_report(all_true_values, all_predictions, target_names=target_names))

    cm = confusion_matrix(all_true_values, all_predictions)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax1_acc = ax1.twinx()
    ax1_acc.plot(val_accuracy, label='Validation Accuracy', color='green', linestyle='--')
    ax1_acc.set_ylabel('Accuracy', color='green')
    ax1_acc.tick_params(axis='y', labelcolor='green')
    ax1_acc.legend(loc='upper right')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=["Decrease", "Increase"], yticklabels=["Decrease", "Increase"])    
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "model/adaptive_model.pth")
    joblib.dump(x_scalar, "model/x_scalar.gz")
