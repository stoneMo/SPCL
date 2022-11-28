# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7


class SPCLCriterion(nn.Module):
    '''
    Args:
        init:
            batch_size (integer): Number of datasamples per batch.

            normalize (bool, optional): Whether to normalise the reprentations.
                (Default: True)

            temperature (float, optional): The temperature parameter of the
                NT_Xent loss. (Default: 1.0)
        forward:
            z_i (Tensor): Reprentation of view 'i'

            z_j (Tensor): Reprentation of view 'j'
    Returns:
        loss (Tensor): NT_Xent loss between z_i and z_j
    '''

    def __init__(self, batch_size, normalize=True, temperature=1.0):
        super(SPCLCriterion, self).__init__()

        self.temperature = temperature
        self.normalize = normalize

        self.register_buffer('labels', torch.zeros(batch_size * 2).long())

        self.register_buffer('mask', torch.ones(
            (batch_size, batch_size), dtype=bool).fill_diagonal_(0))

        self.batch_size = batch_size

    def forward(self, z_i, z_j):


        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|

        Postives:
        Diagonals of ab and ba '\'

        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        if self.normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)

        else:
            z_i_norm = z_i
            z_j_norm = z_j

        bsz = z_i_norm.size(0)

        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature

        # print("logits_aa:", logits_aa)
        # print("logits_bb:", logits_bb)
        # print("logits_ab:", logits_ab)
        # print("logits_ba:", logits_ba)

        # for last batch data 
        if bsz != self.mask.size(0):
            self.labels = torch.zeros(bsz * 2).long().cuda()
            self.mask = torch.ones((bsz, bsz), dtype=bool).fill_diagonal_(0).cuda()

        # Compute Postive Logits
        logits_ab_pos = logits_ab[torch.logical_not(self.mask)]      # shape: [bs]
        logits_ba_pos = logits_ba[torch.logical_not(self.mask)]      # shape: [bs]

        # print("logits_ab_pos:", logits_ab_pos.shape)
        # print("logits_ba_pos:", logits_ba_pos.shape)

        # Compute Negative Logits
        logit_aa_neg = logits_aa[self.mask].reshape(bsz, -1)       # shape: [bs, bs-1]
        logit_bb_neg = logits_bb[self.mask].reshape(bsz, -1)       # shape: [bs, bs-1]
        logit_ab_neg = logits_ab[self.mask].reshape(bsz, -1)       # shape: [bs, bs-1]
        logit_ba_neg = logits_ba[self.mask].reshape(bsz, -1)       # shape: [bs, bs-1]

        # print("logit_aa_neg:", logit_aa_neg.shape)
        # print("logit_bb_neg:", logit_bb_neg.shape)
        # print("logit_ab_neg:", logit_ab_neg.shape)
        # print("logit_ba_neg:", logit_ba_neg.shape)

        # Postive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)      # shape: [bs*2, 1]
        # print("pos:", pos.shape)

        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)           # shape: [bs, (bs-1)*2]
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)           # shape: [bs, (bs-1)*2]

        # print("neg_a:", neg_a.shape)
        # print("neg_b:", neg_b.shape)

        neg = torch.cat((neg_a, neg_b), dim=0)                          # shape: [bs*2, (bs-1)*2]
        # print("neg:", neg.shape)

        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=1)                           # shape: [bs*2, (bs-1)*2+1]
        # print("logits:", logits.shape)

        # print("self.labels:", self.labels.shape)                        # shape: [bs*2]

        loss = F.cross_entropy(logits, self.labels)

        return loss, logits_aa, logits_bb, logits_ab, logits_ba, pos

class DIPCriterion(nn.Module):
    def __init__(self, kld_weight=1, dip_weight=1, lambda_offdiag=10, lambda_diag=5):
        super(DIPCriterion, self).__init__()

        self.kld_weight = kld_weight
        self.dip_weight = dip_weight
        self.lambda_offdiag = lambda_offdiag
        self.lambda_diag = lambda_diag

    def forward(self, mu, log_var):

        # kld_loss = self.kld_loss(mu, log_var)
        kld_loss = 0
        dip_loss = self.dip_loss(mu, log_var, self.lambda_offdiag, self.lambda_diag)
        
        return self.kld_weight*kld_loss, self.dip_weight*dip_loss

    def kld_loss(self, mu, log_var):
        
        return torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    def dip_loss(self, mu, log_var, lambda_offdiag=10, lambda_diag=5):
        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        # Add Variance for DIP Loss II
        cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0) # [D x D]
        # For DIp Loss I
        # cov_z = cov_mu

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                    lambda_diag * torch.sum((cov_diag - 1) ** 2)
        
        return dip_loss

class MetricCriterion(nn.Module):
    def __init__(self, metric_weight, metric_learner, args):
        super(MetricCriterion, self).__init__()

        self.metric_weight = metric_weight
        self.metric_learner = metric_learner

        # print("args.metric_loss:", args.metric_loss)

        if args.metric_loss == "BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.metric_loss == "SCE":
            self.criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=args.num_prototypes)
        elif args.metric_loss == "L1":
            self.criterion = nn.L1Loss()
        elif args.metric_loss == "L2":
            self.criterion = nn.MSELoss()

    def forward(self, h_i, h_j, v_i, v_j, similarity_label):
        
        loss_h_i_j = self.metric_loss(h_i, h_j, similarity_label, SameImage=True)
        loss_v_i_j = self.metric_loss(v_i, v_j, similarity_label, SameImage=True)

        SameImage_loss = loss_h_i_j + loss_v_i_j

        loss_h_i_v_i = self.metric_loss(h_i, v_i, similarity_label, SameImage=False)
        loss_h_i_v_j = self.metric_loss(h_i, v_j, similarity_label, SameImage=False)
        loss_h_j_v_i = self.metric_loss(h_j, v_i, similarity_label, SameImage=False)
        loss_h_j_v_j = self.metric_loss(h_j, v_j, similarity_label, SameImage=False)


        DiffImage_loss = loss_h_i_v_i + loss_h_i_v_j + loss_h_j_v_i + loss_h_j_v_j

        total_loss = SameImage_loss + DiffImage_loss

        return total_loss

    def metric_loss(self, h_i, h_j, similarity_label, SameImage):

        # print("similarity_label:", type(similarity_label))
        # print("similarity_label:", similarity_label)
        # print("SameImage:", SameImage)

        if SameImage:
            label = torch.ones_like(similarity_label)
        else:
            label = similarity_label

        pred_sim = self.metric_learner(torch.abs(h_i-h_j))
        # print("pred_sim:", pred_sim.shape)
        # print("label:", label)
        loss = self.criterion(pred_sim, label)


        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
