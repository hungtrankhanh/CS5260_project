from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models
import datasets
from functions import train, validate, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from torchvision.utils import save_image

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random 


from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


    # import network
    # args.gen_model is TransGAN_8_8_1 for example
    gen_net = eval('models.'+args.gen_model+'.Generator')(args=args).cuda()
    dis_net = eval('models.'+args.dis_model+'.Discriminator')(args=args).cuda()
    gen_net.set_arch(args.arch, cur_stage=2)

    print("The shit!")

    # weight init: Xavier Uniform
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=gpu_ids)
    dis_net = torch.nn.DataParallel(dis_net.to("cuda:0"), device_ids=gpu_ids)
    
    

    # print(gen_net.module.cur_stage)

    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.g_lr, weight_decay=args.wd)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.fid_stat is not None:
        fid_stat = args.fid_stat
    else:
        raise NotImplementedError  # (f"no fid stat for %s"%args.dataset.lower()")
    assert os.path.exists(fid_stat)

    dataset = datasets.ImageDataset(args, cur_img_size=8)
    train_loader = dataset.train

    writer=SummaryWriter()
    writer_dict = {'writer':writer}
    writer_dict["train_global_steps"]=0
    writer_dict["valid_global_steps"]=0

    best = 1e4
    for epoch in range(args.max_epoch):

        train(args, gen_net = gen_net, dis_net = dis_net, gen_optimizer = gen_optimizer, dis_optimizer = dis_optimizer, gen_avg_param = None, train_loader = train_loader,
            epoch = epoch, writer_dict = writer_dict, fixed_z = None, schedulers=[gen_scheduler, dis_scheduler])

        checkpoint = {'epoch':epoch, 'best_fid':best}
        checkpoint['gen_state_dict'] = gen_net.state_dict()
        checkpoint['dis_state_dict'] = dis_net.state_dict()
        score = validate(args, None, fid_stat, epoch, gen_net, writer_dict, clean_dir=True)
        # print these scores, is it really the latest
        print(f"FID score: {score} - best ID score: {best} || @ epoch {epoch}.")
        if epoch == 0 or epoch > 50:
            if score < best:
                save_checkpoint(checkpoint, is_best=(score<best), output_dir=args.output_dir)
                print("Saved Latest Model!")
                best = score

    checkpoint = {'epoch':epoch, 'best_fid':best}
    checkpoint['gen_state_dict'] = gen_net.state_dict()
    checkpoint['dis_state_dict'] = dis_net.state_dict()
    score = validate(args, None, fid_stat, epoch, gen_net, writer_dict, clean_dir=True)
    save_checkpoint(checkpoint, is_best=(score<best), output_dir=args.output_dir)

    #load model
    #
    # best_model = torch.load("best_model")
    # load_epoch = best_model['epoch']
    # best_fid = best_model['best_fid']
    # gen_net.load_state_dict(best_model['gen_state_dict'])
    # dis_net.load_state_dict(best_model['dis_state_dict'])
    # fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (64, args.latent_dim)))
    #
    # fid_score = validate(args, None, fid_stat, load_epoch, gen_net, writer_dict, clean_dir=True)
    # print(f'FID score: {fid_score} || @ epoch {load_epoch}.')
    # fake_imgs = gen_net(fixed_z, load_epoch).detach()
    #
    # batch_num = len(fake_imgs)
    # for j in range(batch_num):
    #     save_image(fake_imgs[j], 'output/gen_image_{}.png'.format(j))

        
if __name__=="__main__":
  main()    
