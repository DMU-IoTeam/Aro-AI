# utils/metrics.py
from sklearn.metrics import classification_report, confusion_matrix
import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Fall'], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / len(loader), report, cm
