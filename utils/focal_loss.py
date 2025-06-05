import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: [B, C], targets: [B]
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather log probabilities of the correct class
        targets = targets.view(-1, 1)
        log_p_t = log_probs.gather(1, targets).squeeze(1)
        p_t = probs.gather(1, targets).squeeze(1)

        # Convert alpha to correct device
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.squeeze())
        else:
            alpha_t = torch.ones_like(p_t)

        loss = -alpha_t * (1 - p_t) ** self.gamma * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
