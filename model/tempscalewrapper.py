import torch
from torch import nn
import torch.nn.functional as F

class TempScaledModel(nn.Module):
    """
    Wrapper For temperature scaling calibration.
    See https://arxiv.org/pdf/1706.04599.pdf
    """
    def __init__(self, base_model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), temperature=1.0, apply_scale=False, apply_softmax=False): # By default this wrapper does nothing.
        super(TempScaledModel, self).__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.tensor([temperature], device=device), requires_grad=False)
        self.apply_scale = apply_scale
        self.apply_softmax = apply_softmax

    def forward(self, *args, **kwargs):
        logits = self.base_model(*args, **kwargs)
        if self.apply_scale:
            logits = logits / self.temperature
        if self.apply_softmax:
            return F.softmax(logits, dim=1)
        else:
            return logits
    
    def enable_temperature_grad(self):
        """Enable gradients for the temperature while disabling for the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.temperature.requires_grad = True

    def enable_base_model_grad(self):
        """Enable gradients for the base model while disabling for the temperature."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.temperature.requires_grad = False
