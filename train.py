# -*- coding: utf-8 -*-
import os
import gc
import math
import time 
import pickle
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.losses import SPCLCriterion, DIPCriterion, MetricCriterion, SCELoss
from optimisers import get_optimiser

from visual import visual_cluster

import clustering
from utils import AverageMeter

def compute_features(dataloader, model, N, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (_, input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True)
        # retrieve the original view
        if args.finetune or args.eval_disent:
            x_i = input_var
            # print("x_i:", x_i.shape)
        else:
            x_i, _ = torch.split(input_var, [3, 3], dim=1)

        aux = model(x_i).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch_size: (i + 1) * args.batch_size] = aux
        else:
            # special treatment for final batch
            features[i * args.batch_size:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features

def cal_percentage_false_negatives(input_label):

    batch_size = input_label.shape[0]

    mask = torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0)

    total_num_negatives = batch_size * (batch_size - 1)

    for i in range(batch_size):
        for j in range(batch_size):
            if input_label[i] != input_label[j]:    # true negatives 
                mask[i,j] = 1
            else:
                mask[i,j] = 0

    num_false_negatives = total_num_negatives - torch.sum(mask).item()

    percentage_false_negatives = num_false_negatives / total_num_negatives

    return percentage_false_negatives, mask

def cal_similarity_pos_neg(logits_aa, logits_bb, logits_ab, logits_ba, mask):

    bsz = mask.shape[0]

    # Compute Postive Logits
    logits_ab_pos = logits_ab[torch.logical_not(mask)]
    logits_ba_pos = logits_ba[torch.logical_not(mask)]

    # Postive Logits over all samples
    pos = torch.cat((logits_ab_pos, logits_ba_pos))

    # print("pos:", pos.shape)

    pos_mean = torch.mean(pos)

    # print('logits_aa:', logits_aa.shape)
    # print('logits_bb:', logits_bb.shape)
    # print('logits_ab:', logits_ab.shape)
    # print('logits_ba:', logits_ba.shape)
    # print("mask:", mask.shape)
    # print("logits_aa[mask]:", logits_aa[mask].shape)

    # Compute Negative Logits
    logit_aa_neg = logits_aa[mask]
    logit_bb_neg = logits_bb[mask]
    logit_ab_neg = logits_ab[mask]
    logit_ba_neg = logits_ba[mask]

    # Negative Logits over all samples
    neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=0)
    neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=0)

    neg = torch.cat((neg_a, neg_b), dim=0)

    # print("neg:", neg.shape)

    neg_mean = torch.mean(neg)

    # Compute False Negative Logits
    mask_fn = mask.fill_diagonal_(1)
    logit_aa_fn_neg = logits_aa[torch.logical_not(mask_fn)]
    logit_bb_fn_neg = logits_bb[torch.logical_not(mask_fn)]
    logit_ab_fn_neg = logits_ab[torch.logical_not(mask_fn)]
    logit_ba_fn_neg = logits_ba[torch.logical_not(mask_fn)]

    # Negative Logits over all samples
    fn_neg_a = torch.cat((logit_aa_fn_neg, logit_ab_fn_neg), dim=0)
    fn_neg_b = torch.cat((logit_ba_fn_neg, logit_bb_fn_neg), dim=0)

    fn_neg = torch.cat((fn_neg_a, fn_neg_b), dim=0)

    # print("fn_neg:", fn_neg.shape)

    fn_neg_mean = torch.mean(fn_neg)

    return pos_mean, neg_mean, fn_neg_mean, pos, neg, fn_neg


def pretrain(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SPCL

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    if args.model == 'resnet18' or args.model == 'resnet34':
        n_channels = 512
    elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
        n_channels = 2048
    else:
        raise NotImplementedError('model not supported: {}'.format(args.model))

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SPCLCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    criterion_dis = DIPCriterion(kld_weight=1, dip_weight=args.dip_weight, lambda_offdiag=10, 
                                lambda_diag=5).cuda()

    if args.clustering_use:
        # clustering algorithm to use
        deepcluster = clustering.__dict__[args.clustering](args.n_classes)
        criterion_ce = nn.CrossEntropyLoss().cuda()
        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum').cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    # log for mini-experiment
    log_dict = dict()

    percentage_fn_epoch = []
    pos_similarity_epoch = []
    neg_similarity_epoch = []
    true_pos_similarity_epoch = []
    false_neg_similarity_epoch = []


    log_all_dict = dict()


    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        if args.clustering_use:
            # get the features for the whole dataset
            features = compute_features(train_dataloader, encoder, len(dataloaders['pretrain'].dataset), args)

            # clustering loss
            clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

            # assign pseudo-labels
            if args.verbose:
                print('Assign pseudo labels')
            pseudo_labels = clustering.arrange_clustering(deepcluster.images_lists)
            pseudo_labels = torch.from_numpy(pseudo_labels).cuda()

        if args.clustering_use and (epoch+1) % args.classifier_init_freq == 0:
            # mlp initialization
            mlp.module.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.module.classifier.weight.data.normal_(0, 0.01)
            mlp.module.classifier.bias.data.zero_()
            mlp.module.classifier.cuda()

            # create an optimizer for the last fc layer
            optimizer_cls = torch.optim.SGD(
                mlp.module.classifier.parameters(),
                lr=args.scaled_learning_rate,
                weight_decay=args.weight_decay,
            )


        # to do 
        # calculate the number of false negatives & the number of true negatives

        percentage_fn_list = []

        # calculate the similarity of samples features from the same class
        pos_similarity_list = []
        neg_similarity_list = []

        # true positive & false negative
        true_pos_similarity_list = []
        false_neg_similarity_list = []
        # calculate the similarity of samples features from different class


        # log all pos, neg, false_negatives

        epoch_all_dict = dict()

        pos_all_list = []
        neg_all_list = []
        fn_all_list = []


        ''' epoch loop '''
        for i, (input_index, inputs, input_label) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)
            input_label = input_label.cuda(non_blocking=True)

            percentage_fn, mask = cal_percentage_false_negatives(input_label)
            percentage_fn_list.append(percentage_fn)

            # Forward pass
            optimiser.zero_grad()
            if args.clustering_use:
                optimizer_cls.zero_grad()

            # retrieve the 2 views
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            # Get the encoder representation
            h_i = encoder(x_i)

            h_j = encoder(x_j)

            # Get the nonlinear transformation of the representation

            if args.dip_proj:
                z_i, mu_i, log_var_i = mlp(h_i)
                z_j, mu_j, log_var_j = mlp(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)
                # cl_loss_mu = criterion(mu_i, mu_j)

                kld_loss_i, dip_loss_i = criterion_dis(mu_i, log_var_i)
                kld_loss_j, dip_loss_j = criterion_dis(mu_j, log_var_j)

                logging.info("cl_loss: {}".format(cl_loss.item()))
                logging.info("dip_loss_i: {}".format(dip_loss_i.item()))
                logging.info("dip_loss_j: {}".format(dip_loss_j.item()))

                loss = cl_loss + dip_loss_i + dip_loss_j

            elif args.clustering_use:
                z_i = mlp(h_i)
                z_j = mlp(h_j)

                y_i_hat = mlp.module.classifier(h_i)
                y_j_hat = mlp.module.classifier(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)

                # CE loss btw y_i_hat and pseudo label 
                # y_i_hat: (B, 10) 
                # pseudo_label: (B, 1)
                # if i < len(train_dataloader) - 1:

                pseudo_label = pseudo_labels[i * args.batch_size: (i + 1) * args.batch_size]

                # if i == np.random.randint(len(train_dataloader)):
                #     cluster_ce_loss = criterion_bce(pseudo_label.float(), input_label.float())
                #     pseudo_label = input_label
                # else:
                #     cluster_ce_loss = torch.Tensor([0.]).cuda()

                # else:
                #     pseudo_label = pseudo_labels[i * args.batch_size:]
                # print("y_i_hat:", y_i_hat.shape)
                # print("pseudo_label:", pseudo_label.shape)
                ce_loss = criterion_ce(y_i_hat, pseudo_label) + criterion_ce(y_j_hat, pseudo_label)

                loss = cl_loss + args.cluster_weight*ce_loss

            else:
                z_i = mlp(h_i)
                z_j = mlp(h_j)
                loss, logits_aa, logits_bb, logits_ab, logits_ba, pos = criterion(z_i, z_j)

            # similarity btw A and views of A
            pos_view = pos.squeeze(1)
            true_pos_mean = torch.mean(pos_view)
            true_pos_similarity_list.append(true_pos_mean.item())            

            # calculate similarity of samples features from class
            pos_mean, neg_mean, false_neg_mean, pos_all, neg_all, fn_neg_all = cal_similarity_pos_neg(logits_aa, logits_bb, logits_ab, logits_ba, mask)

            # similarity btw A and false negative of A
            false_neg_similarity_list.append(false_neg_mean.item())
            pos_similarity_list.append(pos_mean.item())   # pos + false negative
            neg_similarity_list.append(neg_mean.item())   # negative except fn

            # calculate similarity of positive view, negatives, false negatives
            pos_all_list.extend(pos_view.detach().cpu().numpy().tolist())
            neg_all_list.extend(neg_all.detach().cpu().numpy().tolist())
            fn_all_list.extend(fn_neg_all.detach().cpu().numpy().tolist()) 

            loss.backward()

            optimiser.step()
            if args.clustering_use:
                optimizer_cls.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()


        # information for each epoch 
        percentage_fn_epoch.append(np.mean(percentage_fn_list))
        pos_similarity_epoch.append(np.mean(pos_similarity_list))
        neg_similarity_epoch.append(np.mean(neg_similarity_list))

        true_pos_similarity_epoch.append(np.mean(true_pos_similarity_list))
        false_neg_similarity_epoch.append(np.mean(false_neg_similarity_list))

        epoch_all_dict['all_pos_similarity'] = pos_all_list
        epoch_all_dict['all_true_neg_similarity'] = neg_all_list
        epoch_all_dict['all_false_neg_similarity'] = fn_all_list

        log_all_dict[epoch+1] = epoch_all_dict



        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        state = {
            #'args': args,
            'encoder': encoder.state_dict(),
            'mlp': mlp.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch,
        }

        torch.save(state, args.checkpoint_dir)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_pretrain_loss = None  # reset loss

    # log dict
    log_dict['percentage_fn'] = percentage_fn_epoch
    log_dict['pos_similarity'] = pos_similarity_epoch
    log_dict['true_neg_similarity'] = neg_similarity_epoch
    log_dict['true_pos_similarity'] = true_pos_similarity_epoch
    log_dict['false_neg_similarity'] = false_neg_similarity_epoch

    # save log dict 
    f = open(os.path.join(args.miniExp_dir,"mini_exp_log.pkl"),"wb")
    pickle.dump(log_dict,f)
    f.close()

    f_all = open(os.path.join(args.miniExp_dir,"mini_exp_log_all.pkl"),"wb")
    pickle.dump(log_all_dict,f_all)
    f_all.close()

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

def pretrain_cluster(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SPCL

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    if args.model == 'resnet18' or args.model == 'resnet34':
        n_channels = 512
    elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
        n_channels = 2048
    else:
        raise NotImplementedError('model not supported: {}'.format(args.model))

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SPCLCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    criterion_dis = DIPCriterion(kld_weight=1, dip_weight=args.dip_weight, lambda_offdiag=10, 
                                lambda_diag=5).cuda()

    if args.clustering_use:
        # clustering algorithm to use
        deepcluster = clustering.__dict__[args.clustering](args.num_prototypes)
        criterion_ce = nn.CrossEntropyLoss().cuda()
        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum').cuda()
        softmax = nn.Softmax(dim=1).cuda()


    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            # train_dataloader = tqdm(dataloaders['pretrain'])
            train_dataloader = dataloaders['pretrain']
        else:
            train_dataloader = dataloaders['pretrain']

        train_dataloader.dataset.compute_feature = True

        # get the features for the whole dataset
        features = compute_features(train_dataloader, encoder, len(dataloaders['pretrain'].dataset), args)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        
        pseudo_labels = clustering.arrange_clustering(deepcluster.images_lists)
        # pseudo_labels = torch.from_numpy(pseudo_labels).cuda()

        train_dataloader.dataset.labels = pseudo_labels
        train_dataloader.dataset.compute_feature = False

        # train_dataloader.dataset.data = pseudo_labels
        
        # train_dataset = clustering.cluster_assign(deepcluster.images_lists,
        #                                           dataloaders['pretrain'].dataset, args)

        # uniformly sample per target
        # sampler = clustering.UnifLabelSampler(int(args.reassign * len(train_dataset)),
        #                            deepcluster.images_lists)

        # train_dataloader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     num_workers=8,
        #     # sampler=sampler,
        #     pin_memory=True,
        # )

        # set last fully connected layer

        # mlp initialization
        if torch.cuda.device_count() > 1:
            mlp.module.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.module.classifier.weight.data.normal_(0, 0.01)
            mlp.module.classifier.bias.data.zero_()
            mlp.module.classifier.cuda()
            mlp_classifier_parameters = mlp.module.classifier.parameters()
        else:
            mlp.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.classifier.weight.data.normal_(0, 0.01)
            mlp.classifier.bias.data.zero_()
            mlp.classifier.cuda()
            mlp_classifier_parameters = mlp.classifier.parameters()

        # create an optimizer for the last fc layer
        optimizer_cls = torch.optim.SGD(
            mlp_classifier_parameters,
            lr=args.scaled_learning_rate,
            weight_decay=args.weight_decay,
        )


        ''' epoch loop '''
        for i, (input_index, inputs, input_label) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)
            input_label = input_label.cuda(non_blocking=True).squeeze()

            # Forward pass
            optimiser.zero_grad()
            optimizer_cls.zero_grad()

            # retrieve the 2 views
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            # Get the encoder representation
            h_i = encoder(x_i)

            h_j = encoder(x_j)

            # Get the nonlinear transformation of the representation

            if args.dip_proj:
                z_i, mu_i, log_var_i = mlp(h_i)
                z_j, mu_j, log_var_j = mlp(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)
                # cl_loss_mu = criterion(mu_i, mu_j)

                kld_loss_i, dip_loss_i = criterion_dis(mu_i, log_var_i)
                kld_loss_j, dip_loss_j = criterion_dis(mu_j, log_var_j)

                logging.info("cl_loss: {}".format(cl_loss.item()))
                logging.info("dip_loss_i: {}".format(dip_loss_i.item()))
                logging.info("dip_loss_j: {}".format(dip_loss_j.item()))

                loss = cl_loss + dip_loss_i + dip_loss_j

            elif args.clustering_use:
                z_i = mlp(h_i)
                z_j = mlp(h_j)

                if torch.cuda.device_count() > 1:
                    y_i_hat = mlp.module.classifier(h_i)
                    y_j_hat = mlp.module.classifier(h_j)
                
                else:
                    y_i_hat = mlp.classifier(h_i)
                    y_j_hat = mlp.classifier(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)

                # CE loss btw y_i_hat and pseudo label 
                # y_i_hat: (B, 10) 
                # pseudo_label: (B, 1)

                # pseudo_label = pseudo_labels[i * args.batch_size: (i + 1) * args.batch_size]
                
                # print("y_i_hat:", y_i_hat.shape)
                # print("inputs:", inputs.shape)
                # print("pseudo_label:", input_label.shape)

                ce_loss = criterion_ce(y_i_hat, input_label) + criterion_ce(y_j_hat, input_label) 

                # y_i_hat_label = torch.argmax(softmax(y_j_hat),dim=1)
                # y_j_hat_label = torch.argmax(softmax(y_i_hat),dim=1)

                # bce_loss = criterion_ce(y_i_hat, y_i_hat_label) + criterion_ce(y_j_hat, y_j_hat_label) 

                # ce_loss = criterion_ce(y_i_hat, input_label)
                # bce_loss = -torch.mean(torch.sum(softmax(y_i_hat) * torch.log(softmax(y_j_hat)), dim=1))

                loss = cl_loss + args.cluster_weight * ce_loss

            else:
                z_i = mlp(h_i)
                z_j = mlp(h_j)
                loss = criterion(z_i, z_j)

            loss.backward()

            optimiser.step()
            optimizer_cls.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        state = {
            #'args': args,
            'encoder': encoder.state_dict(),
            'mlp': mlp.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch,
        }

        torch.save(state, args.checkpoint_dir)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_pretrain_loss = None  # reset loss

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

def pretrain_cluster_metric(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SPCL

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    if args.model == 'resnet18' or args.model == 'resnet34':
        n_channels = 512
    elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
        n_channels = 2048
    else:
        raise NotImplementedError('model not supported: {}'.format(args.model))

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SPCLCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    criterion_dis = DIPCriterion(kld_weight=1, dip_weight=args.dip_weight, lambda_offdiag=10, 
                                lambda_diag=5).cuda()

    if torch.cuda.device_count() > 1:
        metric_learner = mlp.module.metric_learner
    else:
        metric_learner = mlp.metric_learner

    criterion_metric = MetricCriterion(metric_weight=args.metric_weight, metric_learner=metric_learner, args=args).cuda()

    if args.clustering_use:
        # clustering algorithm to use
        deepcluster = clustering.__dict__[args.clustering](args.num_prototypes)
        if args.cluster_loss == "CE":
            criterion_ce = nn.CrossEntropyLoss().cuda()
        elif args.cluster_loss == "SCE":
            criterion_ce = SCELoss(alpha=0.1, beta=1.0, num_classes=args.num_prototypes).cuda()

        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum').cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    # log for mini-experiment
    log_dict = dict()
    log_all_dict = dict()

    percentage_fn_epoch = []
    pos_similarity_epoch = []
    neg_similarity_epoch = []
    true_pos_similarity_epoch = []
    false_neg_similarity_epoch = []

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            # train_dataloader = tqdm(dataloaders['pretrain'])
            train_dataloader = dataloaders['pretrain']
        else:
            train_dataloader = dataloaders['pretrain']

        train_dataloader.dataset.compute_feature = False         # True or false
        train_dataloader.dataset.pretrain = False         # True or false

        # get the features for the whole dataset
        features = compute_features(train_dataloader, encoder, len(dataloaders['pretrain'].dataset), args)

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        
        pseudo_labels = clustering.arrange_clustering(deepcluster.images_lists)
        
        # tSNE visualization
        if args.visual_cls and (epoch+1) % args.visual_freq == 0:
            visual_cluster(features, pseudo_labels, args, epoch)

        train_dataloader.dataset.num_classes = args.num_prototypes
        train_dataloader.dataset.labels = pseudo_labels

        # train_dataloader.dataset.data_dict = train_dataloader.dataset.loadToMem()
        train_dataloader.dataset.loadToMem()

        train_dataloader.dataset.compute_feature = False
        train_dataloader.dataset.pretrain = True         # True or false

        # train_dataset = clustering.cluster_assign(deepcluster.images_lists,
        #                                           dataloaders['pretrain'].dataset, args)

        # # uniformly sample per target
        # sampler = clustering.UnifLabelSampler(int(args.reassign * len(train_dataset)),
        #                            deepcluster.images_lists)

        # train_dataloader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     num_workers=8,
        #     # sampler=sampler,
        #     pin_memory=True,
        # )

        # set last fully connected layer

        # mlp initialization
        if torch.cuda.device_count() > 1:
            mlp.module.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.module.classifier.weight.data.normal_(0, 0.01)
            mlp.module.classifier.bias.data.zero_()
            mlp.module.classifier.cuda()
            mlp_classifier_parameters = mlp.module.classifier.parameters()
        else:
            mlp.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.classifier.weight.data.normal_(0, 0.01)
            mlp.classifier.bias.data.zero_()
            mlp.classifier.cuda()
            mlp_classifier_parameters = mlp.classifier.parameters()

        # create an optimizer for the last fc layer
        optimizer_cls = torch.optim.SGD(
            mlp_classifier_parameters,
            lr=args.scaled_learning_rate,
            weight_decay=args.weight_decay,
        )

        # percentage of false negatives
        percentage_fn_list = []

        # calculate the similarity of samples features from the same class
        pos_similarity_list = []
        neg_similarity_list = []

        # true positive & false negative
        true_pos_similarity_list = []
        false_neg_similarity_list = []

        # log all pos, neg, false_negatives

        epoch_all_dict = dict()

        pos_all_list = []
        neg_all_list = []
        fn_all_list = []

        ''' epoch loop '''
        for i, (similarity_label, inputs_1, inputs_2, input_label) in enumerate(train_dataloader):

            inputs_1 = inputs_1.cuda(non_blocking=True)
            inputs_2 = inputs_2.cuda(non_blocking=True)
            similarity_label = similarity_label.cuda(non_blocking=True)

            input_label = input_label.cuda(non_blocking=True)

            # calculate the percetange of false negative
            percentage_fn, mask = cal_percentage_false_negatives(input_label[:,0])
            percentage_fn_list.append(percentage_fn)

            # Forward pass
            optimiser.zero_grad()
            optimizer_cls.zero_grad()

            # retrieve the 2 views
            x1_i, x1_j = torch.split(inputs_1, [3, 3], dim=1)
            x2_i, x2_j = torch.split(inputs_2, [3, 3], dim=1)

            # Get the encoder representation
            h_i = encoder(x1_i)
            h_j = encoder(x1_j)
            v_i = encoder(x2_i)
            v_j = encoder(x2_j)

            # Get the nonlinear transformation of the representation

            if args.dip_proj:
                z_i, mu_i, log_var_i = mlp(h_i)
                z_j, mu_j, log_var_j = mlp(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)
                # cl_loss_mu = criterion(mu_i, mu_j)

                kld_loss_i, dip_loss_i = criterion_dis(mu_i, log_var_i)
                kld_loss_j, dip_loss_j = criterion_dis(mu_j, log_var_j)

                logging.info("cl_loss: {}".format(cl_loss.item()))
                logging.info("dip_loss_i: {}".format(dip_loss_i.item()))
                logging.info("dip_loss_j: {}".format(dip_loss_j.item()))

                loss = cl_loss + dip_loss_i + dip_loss_j

            elif args.clustering_use and not args.metric_learn:
                z_i = mlp(h_i)
                z_j = mlp(h_j)

                y_i_hat = mlp.module.classifier(h_i)
                y_j_hat = mlp.module.classifier(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)

                # CE loss btw y_i_hat and pseudo label 
                # y_i_hat: (B, 10) 
                # pseudo_label: (B, 1)
                
                # print("y_i_hat:", y_i_hat.shape)
                # print("inputs:", inputs.shape)
                # print("pseudo_label:", input_label.shape)

                ce_loss = criterion_ce(y_i_hat, input_label) + criterion_ce(y_j_hat, input_label)

                loss = cl_loss + args.cluster_weight*ce_loss

            elif args.clustering_use and args.metric_learn:
                
                # contrastive loss
                z_hi = mlp(h_i)
                z_hj = mlp(h_j)
                z_vi = mlp(v_i)
                z_vj = mlp(v_j)

                loss_h, logits_aa, logits_bb, logits_ab, logits_ba, pos = criterion(z_hi, z_hj)
                loss_v, logits_aa_v, logits_bb_v, logits_ab_v, logits_ba_v, pos_v = criterion(z_vi, z_vj)

                cl_loss = loss_h + loss_v

                # ce loss 
                if torch.cuda.device_count() > 1:
                    y_hi_hat = mlp.module.classifier(h_i)
                    y_hj_hat = mlp.module.classifier(h_j)
                else:
                    y_hi_hat = mlp.classifier(h_i)
                    y_hj_hat = mlp.classifier(h_j)

                y_hmean_hat = (y_hi_hat + y_hj_hat) / 2

                if torch.cuda.device_count() > 1:
                    y_vi_hat = mlp.module.classifier(v_i)
                    y_vj_hat = mlp.module.classifier(v_j)
                else:
                    y_vi_hat = mlp.classifier(v_i)
                    y_vj_hat = mlp.classifier(v_j)

                # y_vmean_hat = (y_vi_hat + y_vj_hat) / 2

                # y_i_hat_label = torch.argmax(softmax(y_j_hat),dim=1)
                # y_j_hat_label = torch.argmax(softmax(y_i_hat),dim=1)

                # bce_loss = criterion_ce(y_i_hat, y_i_hat_label) + criterion_ce(y_j_hat, y_j_hat_label) 

                h_ce_loss = criterion_ce(y_hi_hat, input_label[:,0]) + criterion_ce(y_hj_hat, input_label[:,0])
                v_ce_loss = criterion_ce(y_vi_hat, input_label[:,1]) + criterion_ce(y_vj_hat, input_label[:,1])
                # h_ce_loss = criterion_ce(y_hi_hat, input_label[:,0]) 
                # v_ce_loss = criterion_ce(y_vi_hat, input_label[:,1])
                ce_loss = h_ce_loss + v_ce_loss

                # metric loss
                metric_loss = criterion_metric(h_i, h_j, v_i, v_j, similarity_label)


                loss = args.contras_weight * cl_loss + args.metric_weight * metric_loss + args.cluster_weight * ce_loss

            else:
                z_i = mlp(h_i)
                z_j = mlp(h_j)
                loss = criterion(z_i, z_j)

            # similarity btw A and views of A
            pos_view = pos.squeeze(1)
            true_pos_mean = torch.mean(pos_view)
            true_pos_similarity_list.append(true_pos_mean.item())            

            # calculate similarity of samples features from class
            pos_mean, neg_mean, false_neg_mean, pos_all, neg_all, fn_neg_all = cal_similarity_pos_neg(logits_aa, logits_bb, logits_ab, logits_ba, mask)

            # similarity btw A and false negative of A
            false_neg_similarity_list.append(false_neg_mean.item())
            pos_similarity_list.append(pos_mean.item())   # pos + false negative
            neg_similarity_list.append(neg_mean.item())   # negative except fn

            # calculate similarity of positive view, negatives, false negatives
            pos_all_list.extend(pos_view.detach().cpu().numpy().tolist())
            neg_all_list.extend(neg_all.detach().cpu().numpy().tolist())
            fn_all_list.extend(fn_neg_all.detach().cpu().numpy().tolist()) 

            loss.backward()

            optimiser.step()
            optimizer_cls.step()

            torch.cuda.synchronize()

            sample_count += input_label.size(0)

            run_loss += loss.item()

        # information for each epoch 
        percentage_fn_epoch.append(np.mean(percentage_fn_list))
        pos_similarity_epoch.append(np.mean(pos_similarity_list))
        neg_similarity_epoch.append(np.mean(neg_similarity_list))

        true_pos_similarity_epoch.append(np.mean(true_pos_similarity_list))
        false_neg_similarity_epoch.append(np.mean(false_neg_similarity_list))

        epoch_all_dict['all_pos_similarity'] = pos_all_list
        epoch_all_dict['all_true_neg_similarity'] = neg_all_list
        epoch_all_dict['all_false_neg_similarity'] = fn_all_list

        log_all_dict[epoch+1] = epoch_all_dict

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        state = {
            #'args': args,
            'encoder': encoder.state_dict(),
            'mlp': mlp.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch,
        }

        torch.save(state, args.checkpoint_dir)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_pretrain_loss = None  # reset loss

    # log dict
    log_dict['percentage_fn'] = percentage_fn_epoch
    log_dict['pos_similarity'] = pos_similarity_epoch
    log_dict['true_neg_similarity'] = neg_similarity_epoch
    log_dict['true_pos_similarity'] = true_pos_similarity_epoch
    log_dict['false_neg_similarity'] = false_neg_similarity_epoch

    # save log dict 
    f = open(os.path.join(args.miniExp_dir,"mini_exp_log.pkl"),"wb")
    pickle.dump(log_dict,f)
    f.close()

    f_all = open(os.path.join(args.miniExp_dir,"mini_exp_log_all.pkl"),"wb")
    pickle.dump(log_all_dict,f_all)
    f_all.close()

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

def pretrain_metric(encoder, mlp, dataloaders, args):
    ''' Pretrain script - SPCL

        Pretrain the encoder and projection head with a Contrastive NT_Xent Loss.
    '''

    if args.model == 'resnet18' or args.model == 'resnet34':
        n_channels = 512
    elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
        n_channels = 2048
    else:
        raise NotImplementedError('model not supported: {}'.format(args.model))

    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = SPCLCriterion(batch_size=args.batch_size, normalize=True,
                                temperature=args.temperature).cuda()

    criterion_dis = DIPCriterion(kld_weight=1, dip_weight=args.dip_weight, lambda_offdiag=10, 
                                lambda_diag=5).cuda()

    criterion_metric = MetricCriterion(metric_weight=args.metric_weight, metric_learner=mlp.module.metric_learner, args=args).cuda()

    if args.clustering_use:
        # clustering algorithm to use
        deepcluster = clustering.__dict__[args.clustering](args.n_classes)
        if args.cluster_loss == "CE":
            criterion_ce = nn.CrossEntropyLoss().cuda()
        elif args.cluster_loss == "SCE":
            criterion_ce = SCELoss(alpah=0.1, beta=1.0, num_classes=args.num_prototypes).cuda()

        criterion_bce = nn.BCEWithLogitsLoss(reduction='sum').cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    # log for mini-experiment
    log_dict = dict()
    log_all_dict = dict()

    percentage_fn_epoch = []
    pos_similarity_epoch = []
    neg_similarity_epoch = []
    true_pos_similarity_epoch = []
    false_neg_similarity_epoch = []

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        if args.clustering_use:
            # get the features for the whole dataset
            features = compute_features(train_dataloader, encoder, len(dataloaders['pretrain'].dataset), args)

            # clustering loss
            clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

            # assign pseudo-labels
            if args.verbose:
                print('Assign pseudo labels')
            pseudo_labels = clustering.arrange_clustering(deepcluster.images_lists)
            pseudo_labels = torch.from_numpy(pseudo_labels).cuda()

        if args.clustering_use and (epoch+1) % args.classifier_init_freq == 0:
            # mlp initialization
            mlp.module.classifier = nn.Linear(n_channels, args.n_classes)
            mlp.module.classifier.weight.data.normal_(0, 0.01)
            mlp.module.classifier.bias.data.zero_()
            mlp.module.classifier.cuda()

            # create an optimizer for the last fc layer
            optimizer_cls = torch.optim.SGD(
                mlp.module.classifier.parameters(),
                lr=args.scaled_learning_rate,
                weight_decay=args.weight_decay,
            )

        # percentage of false negatives
        percentage_fn_list = []

        # calculate the similarity of samples features from the same class
        pos_similarity_list = []
        neg_similarity_list = []

        # true positive & false negative
        true_pos_similarity_list = []
        false_neg_similarity_list = []

        # log all pos, neg, false_negatives

        epoch_all_dict = dict()

        pos_all_list = []
        neg_all_list = []
        fn_all_list = []

        ''' epoch loop '''
        for i, (similarity_label, inputs_1, inputs_2, input_label) in enumerate(train_dataloader):

            inputs_1 = inputs_1.cuda(non_blocking=True)
            inputs_2 = inputs_2.cuda(non_blocking=True)
            similarity_label = similarity_label.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()
            if args.clustering_use:
                optimizer_cls.zero_grad()

            # retrieve the 2 views
            x1_i, x1_j = torch.split(inputs_1, [3, 3], dim=1)
            x2_i, x2_j = torch.split(inputs_2, [3, 3], dim=1)

            # Get the encoder representation
            h_i = encoder(x1_i)
            h_j = encoder(x1_j)
            v_i = encoder(x2_i)
            v_j = encoder(x2_j)

            # Get the nonlinear transformation of the representation

            if args.dip_proj:
                z_i, mu_i, log_var_i = mlp(h_i)
                z_j, mu_j, log_var_j = mlp(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)
                # cl_loss_mu = criterion(mu_i, mu_j)

                kld_loss_i, dip_loss_i = criterion_dis(mu_i, log_var_i)
                kld_loss_j, dip_loss_j = criterion_dis(mu_j, log_var_j)

                logging.info("cl_loss: {}".format(cl_loss.item()))
                logging.info("dip_loss_i: {}".format(dip_loss_i.item()))
                logging.info("dip_loss_j: {}".format(dip_loss_j.item()))

                loss = cl_loss + dip_loss_i + dip_loss_j

            elif args.clustering_use:
                z_i = mlp(h_i)
                z_j = mlp(h_j)

                y_i_hat = mlp.module.classifier(h_i)
                y_j_hat = mlp.module.classifier(h_j)

                # Calculate NT_Xent loss
                cl_loss = criterion(z_i, z_j)

                # CE loss btw y_i_hat and pseudo label 
                # y_i_hat: (B, 10) 
                # pseudo_label: (B, 1)
                # if i < len(train_dataloader) - 1:

                pseudo_label = pseudo_labels[i * args.batch_size: (i + 1) * args.batch_size]

                # if i == np.random.randint(len(train_dataloader)) and False:
                #     cluster_ce_loss = criterion_bce(pseudo_label, input_label)
                #     pseudo_label = input_label
                # else:
                #     cluster_ce_loss = torch.Tensor([0.]).cuda()

                # else:
                #     pseudo_label = pseudo_labels[i * args.batch_size:]
                # print("y_i_hat:", y_i_hat.shape)
                # print("pseudo_label:", pseudo_label.shape)
                ce_loss = criterion_ce(y_i_hat, pseudo_label) + criterion_ce(y_j_hat, pseudo_label)

                loss = cl_loss + args.cluster_weight*ce_loss

            elif args.metric_learn:

                z_hi = mlp(h_i)
                z_hj = mlp(h_j)
                z_vi = mlp(v_i)
                z_vj = mlp(v_j)

                loss_h, logits_aa, logits_bb, logits_ab, logits_ba, pos = criterion(z_hi, z_hj)
                loss_v, logits_aa_v, logits_bb_v, logits_ab_v, logits_ba_v, pos_v = criterion(z_vi, z_vj)

                cl_loss = loss_h + loss_v

                metric_loss = criterion_metric(h_i, h_j, v_i, v_j, similarity_label)

                # cl_loss = criterion(z_hi, z_hj) + criterion(z_vi, z_vj)

                loss = cl_loss + args.metric_weight * metric_loss

            else:
                z_i = mlp(h_i)
                z_j = mlp(h_j)
                loss = criterion(z_i, z_j)
                
            # similarity btw A and views of A
            pos_view = pos.squeeze(1)
            true_pos_mean = torch.mean(pos_view)
            true_pos_similarity_list.append(true_pos_mean.item())            

            # calculate similarity of samples features from class
            pos_mean, neg_mean, false_neg_mean, pos_all, neg_all, fn_neg_all = cal_similarity_pos_neg(logits_aa, logits_bb, logits_ab, logits_ba, mask)

            # similarity btw A and false negative of A
            false_neg_similarity_list.append(false_neg_mean.item())
            pos_similarity_list.append(pos_mean.item())   # pos + false negative
            neg_similarity_list.append(neg_mean.item())   # negative except fn

            # calculate similarity of positive view, negatives, false negatives
            pos_all_list.extend(pos_view.detach().cpu().numpy().tolist())
            neg_all_list.extend(neg_all.detach().cpu().numpy().tolist())
            fn_all_list.extend(fn_neg_all.detach().cpu().numpy().tolist()) 

            loss.backward()

            optimiser.step()
            if args.clustering_use:
                optimizer_cls.step()

            torch.cuda.synchronize()

            sample_count += input_label.size(0)

            run_loss += loss.item()

        # information for each epoch 
        percentage_fn_epoch.append(np.mean(percentage_fn_list))
        pos_similarity_epoch.append(np.mean(pos_similarity_list))
        neg_similarity_epoch.append(np.mean(neg_similarity_list))

        true_pos_similarity_epoch.append(np.mean(true_pos_similarity_list))
        false_neg_similarity_epoch.append(np.mean(false_neg_similarity_list))

        epoch_all_dict['all_pos_similarity'] = pos_all_list
        epoch_all_dict['all_true_neg_similarity'] = neg_all_list
        epoch_all_dict['all_false_neg_similarity'] = fn_all_list

        log_all_dict[epoch+1] = epoch_all_dict

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        state = {
            #'args': args,
            'encoder': encoder.state_dict(),
            'mlp': mlp.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch,
        }

        torch.save(state, args.checkpoint_dir)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_pretrain_loss = None  # reset loss

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def supervised(encoder, mlp, dataloaders, args):
    ''' Supervised Train script - SPCL

        Supervised Training encoder and train the supervised classification head with a Cross Entropy Loss.
    '''

    mode = 'pretrain'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((encoder, mlp), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = (1e-12 / args.warmup_epochs) * args.learning_rate

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, args.n_epochs, eta_min=0.0, last_epoch=-1)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()
        mlp.train()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_top5 = 0.0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (_, inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            h = encoder(inputs)

            # Take pretrained encoder representations
            output = mlp(h)

            loss = criterion(output, target)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)

            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            _, output_topk = output.topk(5, 1, True, True)

            acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_top5 += acc_top5

        epoch_pretrain_loss = run_loss / len(dataloaders['train'].dataset)  # sample_count

        epoch_pretrain_acc = run_top1 / len(dataloaders['train'].dataset)

        epoch_pretrain_acc_top5 = run_top5 / len(dataloaders['train'].dataset)

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (float(epoch+1) / args.warmup_epochs) * args.learning_rate
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {
                                    'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('supervised_epoch_acc', {
                                    'pretrain': epoch_pretrain_acc}, epoch+1)
            args.writer.add_scalars('supervised_epoch_acc_top5', {
                                    'pretrain': epoch_pretrain_acc_top5}, epoch+1)
            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        state = {
            #'args': args,
            'encoder': encoder.state_dict(),
            'mlp': mlp.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch,
        }

        torch.save(state, args.checkpoint_dir)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_pretrain_loss = None  # reset loss
    
    # log dict
    log_dict['percentage_fn'] = percentage_fn_epoch
    log_dict['pos_similarity'] = pos_similarity_epoch
    log_dict['true_neg_similarity'] = neg_similarity_epoch
    log_dict['true_pos_similarity'] = true_pos_similarity_epoch
    log_dict['false_neg_similarity'] = false_neg_similarity_epoch

    # save log dict 
    f = open(os.path.join(args.miniExp_dir,"mini_exp_log.pkl"),"wb")
    pickle.dump(log_dict,f)
    f.close()

    f_all = open(os.path.join(args.miniExp_dir,"mini_exp_log_all.pkl"),"wb")
    pickle.dump(log_all_dict,f_all)
    f_all.close()

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def finetune(encoder, mlp, dataloaders, args):
    ''' Finetune script - SPCL

        Freeze the encoder and train the supervised classification head with a Cross Entropy Loss.
    '''

    mode = 'finetune'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((mlp,), mode, args)

    ''' Schedulers '''
    # Cosine LR Decay
    lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.finetune_epochs)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    best_valid_acc = 0.0
    patience_counter = 0

    ''' Finetune loop '''
    for epoch in range(args.finetune_epochs):

        # Freeze the encoder, train classification head
        encoder.eval()
        mlp.train()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_topk = 0.0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.finetune_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (_, inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            # Do not compute the gradients for the frozen encoder
            with torch.no_grad():
                h = encoder(inputs)

            # Take pretrained encoder representations
            output = mlp(h)

            loss = criterion(output, target)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)

            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            # add top ceil(k/2) accuracy 
            top_k = math.ceil(args.n_classes / 2) 
            if top_k > 5:
                top_k = 5
            _, output_topk = output.topk(top_k, 1, True, True)

            acc_topk = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_topk += acc_topk

        epoch_finetune_loss = run_loss / len(dataloaders['train'])  # sample_count

        epoch_finetune_acc = run_top1 / len(dataloaders['train'])

        epoch_finetune_acc_topk = run_topk / len(dataloaders['train'])

        ''' Update Schedulers '''
        # Decay lr with CosineAnnealingLR
        lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Finetune] loss: {:.4f},\t acc: {:.4f}, \t acc_top_{}: {:.4f}\n'.format(
                epoch_finetune_loss, epoch_finetune_acc, top_k, epoch_finetune_acc_topk))

            args.writer.add_scalars('finetune_epoch_loss', {'train': epoch_finetune_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {'train': epoch_finetune_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top_'+str(top_k), {
                                    'train': epoch_finetune_acc_topk}, epoch+1)
            args.writer.add_scalars(
                'finetune_lr', {'train': optimiser.param_groups[0]['lr']}, epoch+1)

        valid_loss, valid_acc, epoch_valid_acc_topk = evaluate(
            encoder, mlp, dataloaders, 'valid', epoch, args)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if valid_acc >= best_valid_acc:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_acc = valid_acc

            # saving using process (rank) 0 only as all processes are in sync

            state = {
                #'args': args,
                'encoder': encoder.state_dict(),
                'supp_mlp': mlp.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch
            }

            torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))
        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                # break

        epoch_finetune_loss = None  # reset loss
        epoch_finetune_acc = None
        epoch_finetune_acc_topk = None

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

def evaluate(encoder, mlp, dataloaders, mode, epoch, args):
    ''' Evaluate script - SPCL

        evaluate the encoder and classification head with Cross Entropy loss.
    '''

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc
    epoch_valid_acc_topk = None

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()
    mlp.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_topk = 0.0

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    ''' epoch loop '''
    for i, (_, inputs, target) in enumerate(eval_dataloader):

        # Do not compute gradient for encoder and classification head
        encoder.zero_grad()
        mlp.zero_grad()

        inputs = inputs.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # Forward pass

        h = encoder(inputs)

        output = mlp(h)

        loss = criterion(output, target)

        torch.cuda.synchronize()

        sample_count += inputs.size(0)

        run_loss += loss.item()

        predicted = output.argmax(-1)

        acc = (predicted == target).sum().item() / target.size(0)

        run_top1 += acc

        # add top ceil(k/2) accuracy 
        top_k = math.ceil(args.n_classes / 2) 
        if top_k > 5:
            top_k = 5
        _, output_topk = output.topk(top_k, 1, True, True)

        acc_topk = (output_topk == target.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / target.size(0)  # num corrects

        run_topk += acc_topk

    epoch_valid_loss = run_loss / len(dataloaders[mode])  # sample_count

    epoch_valid_acc = run_top1 / len(dataloaders[mode])

    epoch_valid_acc_topk = run_topk / len(dataloaders[mode])

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[{}] loss: {:.4f},\t acc: {:.4f},\t acc_top_{}: {:.4f} \n'.format(
            mode, epoch_valid_loss, epoch_valid_acc, top_k, epoch_valid_acc_topk))

        if mode != 'test':
            args.writer.add_scalars('finetune_epoch_loss', {mode: epoch_valid_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {mode: epoch_valid_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top_'+str(top_k), {
                                    'train': epoch_valid_acc_topk}, epoch+1)

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

    return epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_topk

def eval_disent(encoder, dataloaders, args):
    ''' 

        evaluate the disentanglement of the learned representation.
    '''

    # epoch_valid_loss = None  # reset loss
    # epoch_valid_acc = None  # reset acc
    # epoch_valid_acc_topk = None

    # ''' Loss / Criterion '''
    # criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        train_dataloader = tqdm(dataloaders['train'])
        valid_dataloader = tqdm(dataloaders['valid'])
        test_dataloader = tqdm(dataloaders['test'])
    else:
        train_dataloader = dataloaders['train']
        valid_dataloader = dataloaders['valid']
        test_dataloader = dataloaders['test']

    # get the features for the training dataset
    train_features = compute_features(train_dataloader, encoder, len(dataloaders['train'].dataset), args)
    valid_features = compute_features(valid_dataloader, encoder, len(dataloaders['valid'].dataset), args)
    test_features = compute_features(test_dataloader, encoder, len(dataloaders['test'].dataset), args)

    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    
    feature_dim = train_features.shape[1]

    run_time = 10

    for i in range(run_time):

        start_index = np.random.choice(feature_dim//2, 1)[0]

        logging.info("run_time: {}, start_index: {}".format(i+1, start_index))
        
        if args.debug:
            train_sample = 32
            valid_sample = 32
            test_sample = 32
        else:
            train_sample = train_features.shape[0]
            valid_sample = valid_features.shape[0]
            test_sample = test_features.shape[0]

        
        train_X = train_features[:train_sample,start_index:feature_dim//2+start_index]
        train_y = dataloaders['train'].dataset.labels[:train_sample].numpy()

        print(type(train_X), train_X.shape)
        print(type(train_y), train_y.shape)

        clf.fit(train_X, train_y)


        valid_X = valid_features[:valid_sample,start_index:feature_dim//2+start_index]
        valid_y = dataloaders['valid'].dataset.labels[:valid_sample].numpy()

        valid_predicted = clf.predict(valid_X)

        # print("valid_predicted:", type(valid_predicted))
        # print("valid_y:", type(valid_y))

        valid_acc = (valid_predicted == valid_y).sum().item() / valid_sample


        test_X = test_features[:test_sample,start_index:feature_dim//2+start_index]
        test_y = dataloaders['test'].dataset.labels[:test_sample].numpy()

        test_predicted = clf.predict(test_X)

        test_acc = (test_predicted == test_y).sum().item() / test_sample


        if args.print_progress:  # only validate using process 0
            logging.info('\n[SVM] valid acc: {:.4f} \t test acc: {:.4f} \n'.format(valid_acc, test_acc))

    gc.collect()  # release unreferenced memory

    return valid_acc, test_acc

