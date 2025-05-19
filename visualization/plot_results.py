import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_training(train_losses, val_losses, train_f1s, val_f1s, lrs, save_path="results/training_metrics.png"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Val Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].legend()

    axs[1].plot(train_f1s, label='Train F1')
    axs[1].plot(val_f1s, label='Val F1')
    axs[1].set_title('F1 Score over Epochs')
    axs[1].legend()

    axs[2].plot(lrs, label='Learning Rate')
    axs[2].set_title('Learning Rate over Epochs')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)  # 📸 저장
    plt.close()


def plot_confusion(cm, save_path="results/confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fall"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(save_path)  # 📸 저장
    plt.close()
