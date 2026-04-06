import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss.fldcf_loss.GAN_Loss import GANLoss
from torch.nn.modules.loss import _Loss
from collections import namedtuple


class Loss_fake():
    def __init__(self, ):
        self.criterion = CrossEntropy2d(ignore_label=255).cuda()  # Ignore label ??
        self.gan = GANLoss().cuda()
        self.fake = nn.CrossEntropyLoss().cuda()
        self.criterion2 = torch.nn.BCELoss().cuda()
        self.SoftDice = SoftDiceLoss().cuda()
        self.interp = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.Predictions = namedtuple('predictions', ['vote', 'before_softmax',
                                                      'after_softmax', 'raw'])
        self.softmax = torch.nn.Softmax(dim=1)
        self.image_loss_function = nn.L1Loss()

    def loss_calc(self, out, label, out_label, model):
        out_label = out_label.type(torch.long)
        if (model == 'fldcf' or model == 'mflnet'):
            pred, real_or = out
            b, c, w, h = pred.size()
            label = Variable(label.long()).cuda()

            print("=== loss_calc DEBUG ===")
            print("pred.shape      :", pred.shape)
            print("label.shape     :", label.shape)
            print("real_or.shape   :", real_or.shape)
            print("label.dtype     :", label.dtype)
            print("label.unique    :", torch.unique(label))
            print("out_label.shape :", out_label.shape)
            print("out_label       :", out_label)

            corss = self.criterion(pred, label)
            if (model == 'fldcf'):
                gan = self.fake(real_or, out_label)
                loss = corss + gan
            else:
                loss = corss
        if (model == 'restore'):
            img, _ = out
            loss = self.image_loss_function(img, label)

        return loss


class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    # weight = torch.tensor([2.0, 1.0]).cuda()
    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # loss = F.cross_entropy(predict, target, weight=weight, reduction='elementwise_mean')
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        return dice_loss
