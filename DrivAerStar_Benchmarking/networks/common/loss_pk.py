
import torch
import torch.nn as nn

class RL2Loss(nn.Module):
    def __init__(self,eps=1e-8,reduction='mean'):
        super(RL2Loss, self).__init__()
        self.eps = eps
        self.reduction = eps

    def forward(self, x, y):
        """
        Calculate the relative L2 loss between predictions x and targets y using torch.norm.
        
        Args:
        x (torch.Tensor): Predicted outputs.
        y (torch.Tensor): Ground truth values.
        
        Returns:
        torch.Tensor: Computed relative L2 loss.
        """
        # Calculate the numerator and denominator using torch.norm
        num = torch.norm(x - y)
        denom = torch.norm(y)
        # Avoid division by zero
        relative_loss = num / (denom + self.eps)
        return relative_loss


class MRE(nn.Module):
    def __init__(self, eps=1e-8, reduction='mean'):
        super(MRE, self).__init__()
        self.eps = eps
        self.reduction = eps

    def forward(self, output, target):

        if target.shape != output.shape:
            target = target.unsqueeze(-1)
        error = output - target

        norm_error_sample = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-8)
        norm_error_channel = norm_error_sample.reshape(-1)

        # norm_error_channel = torch.mean(norm_error_sample, dim=-1)
        norm_error_batch = torch.mean(norm_error_channel)

        return norm_error_batch



class Rating(object):
    def __init__(self, ):
        super(Rating, self).__init__()
        self.criterion_1 = nn.MSELoss()
        self.criterion_2 = nn.L1Loss()

    def __call__(self, x, y):
        # print("MAE",self.criterion_2(x, y))
        # print("RMSE",torch.sqrt(self.criterion_1(x, y)))
        return 0.4/self.criterion_2(x, y) + 0.4/torch.sqrt(self.criterion_1(x, y))


class HarmonicMeanRatingLoss(object):
    def __init__(self, ):
        super(HarmonicMeanRatingLoss, self).__init__()
        self.loss = Rating()

    def __call__(self, x, y):
        return 1/self.loss(x, y)
