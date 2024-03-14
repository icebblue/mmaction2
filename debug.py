import torch
import torch.nn as nn
import torch.nn.functional as F

class JSDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(JSDivergence, self).__init__()
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        # if self.teacher_detach:
        #     preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T, dim=1)
        softmax_pred_S = F.softmax(preds_S, dim=1)
        log_mean_pred = ((softmax_pred_T + softmax_pred_S) * 0.5).log()
        loss = (F.kl_div(log_mean_pred, softmax_pred_S, reduction=self.reduction) + 
                F.kl_div(log_mean_pred, softmax_pred_T, reduction=self.reduction)) * 0.5
        return self.loss_weight * loss
    


if __name__ == "__main__":
    js_div = JSDivergence()
    t1 = torch.randn(1,1001)
    print(js_div(t1, t1))

