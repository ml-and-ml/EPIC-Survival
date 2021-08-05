import torch
import torch.nn as nn
import torch.nn.functional as F

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = F._Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class NLPLLossStrat(_Loss):
    def __init__(self):
        super(NLPLLossStrat, self).__init__()

    def forward(self, risk_pred, y, e, embedding, target, low, high, gpu, args):
        e = e.int().unsqueeze(1).to(gpu)
        y= y.unsqueeze(1)
        mask = torch.ones(y.shape[0], y.shape[0], device=gpu)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        clustering_loss = F.mse_loss(embedding, target.cuda())
        strat_loss = 1 / (1 + torch.abs((high.mean() - low.mean())))
        strat_loss = F.smooth_l1_loss(strat_loss, torch.zeros(1).squeeze().to(gpu), reduction='none').to(gpu)
        return neg_log_loss + args.clusteringlambda * clustering_loss + strat_loss

