import torch

class CFG:
    epochs = 50
    batch_size = 32
    lr = 1e-3
    patience = 5
    warmup_ratio = 0.1
    dropout = 0.3

    train_fall_dir = "reduced_extracted_data/npy/train/fall"
    train_normal_dir = "reduced_extracted_data/npy/train/normal"
    val_fall_dir = "reduced_extracted_data/npy/val/fall"
    val_normal_dir = "reduced_extracted_data/npy/val/normal"

    save_path = "reduced_fall_gru_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"