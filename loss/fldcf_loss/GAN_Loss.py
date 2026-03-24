import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GANLoss(nn.Module):
    def __init__(self, gan_type='lsgan', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss #adicionar el ()
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target):
        # print("shape TARGET IS REAL: ", target_is_real)
        #target_label = self.get_target_label(input, target_is_real)
        #####
        # print("TYPE: input SAVE",np.save("/work-nfs/lsalgueiro/git/mnt/BasicSR/debugging+"  ,input.numpy()))
        # print("shape INPUT: ", input)
        # print("shape TARGET: ", target_label)

        ######
        loss = self.loss(input, target)
        # print("[GANloss...] FORWARD LOSS: ",loss)
        return loss