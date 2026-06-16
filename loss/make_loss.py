# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcontrast import ImgToProConLoss


def make_loss(cfg, num_classes, device):    # modified by gu
    sampler = cfg.DATALOADER.TRAIN.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            # print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            # print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    # else:
    #     print('expected METRIC_LOSS_TYPE should be triplet'
    #           'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'RandomIdentitySampler':
        i2p_contrast_loss = ImgToProConLoss(device)
        def loss_func(score, feat, target, target_cam, i2tscore = None, t_prototypes = None, batch_aids = None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if t_prototypes is not None: 
                        prototypes_labels = torch.arange(t_prototypes.size(0)).to(device)
                        I2P_LOSS = i2p_contrast_loss(image_features = feat[-1], 
                                                     text_prototypes = t_prototypes, 
                                                     i_labels = target, 
                                                     p_labels = batch_aids)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2P_LOSS + loss

                    # if i2tscore != None:
                    #     I2TLOSS = xent(i2tscore, target)
                    #     loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                    if not torch.is_tensor(loss):
                        base = score[0] if isinstance(score, list) else score
                        loss = torch.as_tensor(loss, dtype=base.dtype, device=base.device)
                    return loss, ID_LOSS, TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    if not torch.is_tensor(loss):
                        base = score[0] if isinstance(score, list) else score
                        loss = torch.as_tensor(loss, dtype=base.dtype, device=base.device)
                    return loss, ID_LOSS, TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, or RandomIdentitySampler'
              'but got {}'.format(cfg.DATALOADER.TRAIN.SAMPLER))
    return loss_func, center_criterion


