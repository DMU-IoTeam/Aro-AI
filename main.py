import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import CFG
from dataset.fall_dataset import get_dataloaders
from model.gru_model import FallGRUClassifier
from utils.scheduler import get_scheduler
from utils.metrics import evaluate
from utils.focal_loss import FocalLoss
from visualization.plot_results import plot_training, plot_confusion


# 📁 결과 저장 폴더 생성
os.makedirs("reduced_results", exist_ok=True)

# ✅ 데이터 로딩
train_loader, val_loader = get_dataloaders(CFG)

# ✅ 모델 & 설정
model = FallGRUClassifier().to(CFG.device)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
total_steps = len(train_loader) * CFG.epochs
scheduler = get_scheduler(optimizer, total_steps, int(CFG.warmup_ratio * total_steps))
# weights = torch.tensor([1.0, 1.5]).to(CFG.device)
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = FocalLoss(gamma=2.0, alpha=[1.0, 1.5]).to(CFG.device)

# ✅ 기록용 리스트
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []
lrs = []

# ✅ 학습 루프
best_f1, wait = 0.0, 0
for epoch in range(CFG.epochs):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    preds, targets = [], []

    for x, y in train_loader:
        x, y = x.to(CFG.device), y.to(CFG.device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        pred = out.argmax(dim=1)
        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    # 📊 평가
    train_preds = torch.tensor(preds)
    train_targets = torch.tensor(targets)
    train_acc = (train_preds == train_targets).float().mean().item()
    train_recall = (train_preds[train_targets == 1] == 1).float().mean().item()
    train_f1 = 2 * train_recall * train_acc / (train_recall + train_acc + 1e-8)

    train_losses.append(epoch_loss / len(train_loader))
    train_f1s.append(train_f1)
    lrs.append(optimizer.param_groups[0]["lr"])

    val_loss, val_report, cm = evaluate(model, val_loader, criterion, CFG.device)
    val_losses.append(val_loss)
    val_f1 = float(val_report.split("avg")[-1].split()[-2])  # weighted avg F1
    val_f1s.append(val_f1)

    # 📋 로그 출력
    print(f"\n📘 Epoch {epoch+1}")
    print(f"Train | Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2%}, Recall: {train_recall:.2%}, F1: {train_f1:.2%}")
    print(f"Val   | Loss: {val_loss:.4f}, F1: {val_f1:.2%}")
    print(val_report)

    # 📝 저장
    with open("reduced_results/val_report_epoch_last.txt", "w") as f:
        f.write(val_report)

    # ✅ Best 모델 저장
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), CFG.save_path)
        with open("reduced_results/val_report_best.txt", "w") as f:
            f.write(val_report)
        print(f"✅ Best model saved (F1: {val_f1:.2%})")
        wait = 0
    else:
        wait += 1
        if wait >= CFG.patience:
            print("⏹️ Early stopping triggered.")
            break

# ✅ 시각화 저장
plot_training(train_losses, val_losses, train_f1s, val_f1s, lrs, save_path="reduced_results/training_metrics.png")
plot_confusion(cm, save_path="reduced_results/confusion_matrix.png")
