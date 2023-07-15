# -*- coding: utf-8 -*-
import argparse
import os
#from apex import amp
import time
import cv2
import torchvision
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import os.path as osp
import datetime
import numpy as np
import torch, gc
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint
from utils import *
from logger import *
from configs.yml_parser import YAMLParser

from model.SpikeFormer_re import SpikeFormer
from Metrics.Metrics import Metrics
from model import Loss_rf as Loss
from datasets.h5_loader_rssf import *
import warnings
warnings.filterwarnings('ignore')
from utils import SaveModel, LoadModel
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--configs', '-c', type=str, default='./configs/spike2flow.yml')
parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--batch_size', '-bs', type=int, default=2)
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--start-epoch', '-se', type=int, default=0)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--print_freq', '-pf', type=int, default=None)
parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--model_iters', '-mit', type=int, default=8)
parser.add_argument('--no_warm', '-nw', action='store_true', default=False)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--warm_iters', '-wi', type=int, default=3000)
parser.add_argument('--eval_vis', '-ev', type=str, default='eval_vis')
parser.add_argument('--crop_len', '-clen', type=int, default=200)
parser.add_argument('--with_valid', '-wv', type=bool, default=True)
parser.add_argument('--decay_interval', '-di', type=int, default=10)
parser.add_argument('--decay_factor', '-df', type=float, default=0.7)
parser.add_argument('--valid_vis_freq', '-vvf', type=float, default=10)
args = parser.parse_args()

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

n_iter = 0
spikeRadius = 32  # half length of input spike sequence expcept for the middle frame
spikeLen = 2 * spikeRadius + 1  # length of input spike sequence
avg_l2_loss = 0.
avg_vgg_loss = 0.
avg_edge_loss = 0.
avg_total_loss = 0.
l2loss = Loss.CharbonnierLoss()
#vggloss = Loss.VGGLoss4('/home/SpikeFormer/vgg19-low-level4.pth').cpu()
criterion_edge = Loss.EdgeLoss()
LAMBDA_L2 = 100.0
LAMBDA_VGG = 1.0
LAMBDA_EDGE = 50.0       

if args.print_freq != None:
    cfg['train']['print_freq'] = args.print_freq
if args.batch_size != None:
    cfg['loader']['batch_size'] = args.batch_size


############################# updata the root ###############################
cfg['data']['spike_path'] = osp.join(cfg['data']['rssf_path'], 'spike')
cfg['data']['dsft_path']  = osp.join(cfg['data']['rssf_path'], 'dsft' )
cfg['data']['flow1_path'] = osp.join(cfg['data']['rssf_path'], 'flow1')
# cfg['data']['flow2_path'] = osp.join(cfg['data']['rssf_path'], 'flow2')
# cfg['data']['flow3_path'] = osp.join(cfg['data']['rssf_path'], 'flow3')
cfg['data']['image_path'] = osp.join(cfg['data']['rssf_path'], 'imgs')
#############################################################################
   
################################# warm up ###################################
warmup = WarmUp(ed_it=args.warm_iters, st_lr=1e-7, ed_lr=args.learning_rate)
def tensriza(n):
    return torch.tensor([item.cpu().detach().numpy() for item in n]).cpu()

##################################################################################################
## Train
def train(cfg, train_loader, model, optimizer, epoch,  train_writer, perIter = 20):
    ######################################################################
    
    ## Init
    global n_iter
    avg_l2_loss = 0.
    avg_vgg_loss = 0.
    avg_edge_loss = 0.
    avg_total_loss = 0.

    #model.train()


    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        # if (not args.no_warm) and (n_iter <= args.warm_iters):
        #     warmup.adjust_lr(optimizer=optimizer, cur_it=n_iter)

        spikes = data['spikes']
        img_gt = data['imgs']
        #Ds = data['dsft']

        spikes = [spk.cpu() for spk in spikes]
        flow1gt = data['flows'][0].cpu()
        flowgt = [flow1gt]#[flow1gt, flow2gt,flow3gt]
        #print('s',spikes.shape)
        spk = torch.cat(spikes, dim =1)
        print(spk.shape)
        flowgt = torch.cat(flowgt, dim=1)

        img_gt = tensriza(img_gt).squeeze(0)#cat(img_gt, dim=1)#diff torch.Size([1, 1, 320, 448]) torch.Size([1, 3600, 1280, 3])
        #flow=  tensriza(flow) #, tensriza(images)

        ## compute loss
        # flowgt=tensriza(flowgt)
        print('training begins!')
        img_s = model(spk).cpu()
        #img_s = model(flowgt).cpu()

        print('v fg', img_s.shape, img_gt.shape, flowgt.shape) 
        # v torch.Size([1, 6, 320, 448]) !!torch.Size([1, 1, 320, 448])!!
        #v torch.Size([3, 1, 2, 320, 448])(when flowgt unconcatenated) torch.Size([1, 1, 320, 448])flio
        

        loss_l2 = l2loss(img_s, flowgt) * LAMBDA_L2
        print('L2 LOSS',loss_l2.detach().cpu().item())
        gc.collect()
        torch.cuda.empty_cache()

        loss_edge = criterion_edge(img_s, img_gt) * LAMBDA_EDGE
        #loss_vgg = vggloss(img_s, img_gt) * LAMBDA_VGG
        del img_gt, img_s, flow1gt, flowgt    


        totalLoss =  loss_l2 +loss_edge

        ## compute gradient and optimize
        optimizer.zero_grad()
        # with amp.scale_loss(totalLoss, optimizer) as scaled_loss:
        totalLoss.backward()

        optimizer.step()


        # avg_l2_loss += loss_l2.detach().cpu()
        # avg_vgg_loss += loss_vgg.detach().cpu()
        # avg_edge_loss += loss_edge.detach().cpu()
        avg_total_loss += totalLoss.detach().cpu()
        if (ww + 1) % perIter == 0:
                # avg_l2_loss = avg_l2_loss / perIter
                # avg_vgg_loss = avg_vgg_loss / perIter
                # avg_edge_loss = avg_edge_loss / perIter
                avg_total_loss = avg_total_loss / perIter
                print('Epoch: %s, Iter: %s' % (epoch, ww + 1))
                print('TotalLoss: %s' % (
                    avg_total_loss.item()))
                # print('L2Loss: %s; VggLoss: %s; EdgeLoss: %s; TotalLoss: %s' % (
                #     avg_l2_loss.item(), avg_vgg_loss.item(), avg_edge_loss.item(), avg_total_loss.item()))
                # avg_l2_loss = 0.
                # avg_vgg_loss = 0.
                # avg_edge_loss = 0.
                avg_total_loss = 0.
        
        n_iter += 1
        if ww>40:
            break

    return


#########################################################################################
## valid
def validation(epoch,cfg, test_datasets, model, saveRoot='/ckpt'):
    global n_iter

    # switch to evaluate mode
    model.eval()
    print('Eval Epoch: %s' %(epoch))
    i_set = 0
    with torch.no_grad():
        pres_a = []
        gts_a = []
        for scene, cur_test_set in test_datasets.items():
            i_set += 1
            cur_test_loader = torch.utils.data.DataLoader(
                cur_test_set,
                pin_memory = False,
                drop_last = False,
                batch_size = 1,
                shuffle = False,
                num_workers = args.num_workers)
            cur_eval_vis_path = osp.join(args.eval_vis, scene)
            make_dir(cur_eval_vis_path)
            i=0
            for ww, data in enumerate(cur_test_loader, 0):
                spikes = data['spikes']
                pres = []
                gts = []
                i+=1
                images = data['imgs']
                spikes = [spk.cpu() for spk in spikes]
                flogt = tensriza(data['flows'][0]).permute(0,3,1,2)
                images = tensriza(images)
                #flogt = [data['flows'][0].cpu()]
                spks = torch.cat(spikes, dim =1)

                # spikes = spks.cpu()
                gtImg = images[-1,:,:,:,0:2].permute(0,3,1,2)
                predImg = model(spks).cpu()
                print('ss',gtImg.shape, predImg.shape)
                # predImg = predImg.squeeze(1)
                # predImg = predImg[:,3:-3,:]

                predImg = predImg.clamp(min=-1., max=1.)
                predImg = predImg.detach().cpu().numpy()
                gtImg = gtImg.clamp(min=-1., max=1.)
                gtImg = gtImg.detach().cpu().numpy()

                predImg = (predImg + 1.) / 2. * 255.
                predImg = predImg.astype(np.uint8)

                gtImg = (gtImg + 1.) / 2. * 255.
                gtImg = gtImg.astype(np.uint8)

                pres.append(predImg)
                gts.append(gtImg)
                pres_a.append(predImg)
                gts_a.append(gtImg)
                # pres, gts  = np.array(pres), np.array(gts)
                pres_e = np.concatenate(pres, axis=0)
                gts_e = np.concatenate(gts, axis=0)
                pres_e = np.concatenate(pres_e, axis=0)
                gts_e = np.concatenate(gts_e, axis=0)
                print('single esti')
                psnr = metrics.Cal_PSNR(pres_e, gts_e)
                ssim = metrics.Cal_SSIM(pres_e, gts_e)
                best_psnr, best_ssim, _ = metrics.GetBestMetrics()
                if psnr >= best_psnr and ssim >= best_ssim:
                    metrics.Update(psnr, ssim)
                if i>10:
                    break
        # pres = np.concatenate(pres, axis=0)
        # gts = np.concatenate(gts, axis=0)

        # psnr = metrics.Cal_PSNR(pres, gts)
        # ssim = metrics.Cal_SSIM(pres, gts)
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()
        from utils import SaveModel, LoadModel
        

        SaveModel(epoch, (best_psnr, best_ssim), model, optimizer, saveRoot)
        # if psnr >= best_psnr and ssim >= best_ssim:
        #     metrics.Update(psnr, ssim)
        SaveModel(epoch, (best_psnr, best_ssim), model, optimizer, saveRoot, best=True)
        with open(f'./{saveRoot}eval_best_log.txt', 'w') as f:
            f.write('epoch: %s; psnr: %s, ssim: %s\n' %(epoch, best_psnr, best_ssim))
        _,B, H, W = predImg.shape#192 256
        divide_line = np.zeros((1,4,W)).astype(np.uint8)
        num = 0
        cdg = Image.fromarray(cv2.cvtColor(gts_a[0][0].transpose(1, 2, 0), cv2.COLOR_BGR2RGB))#Image.fromarray(concatImg.transpose(1, 2, 0)).convert('RGB')
        cdg.save(f'{saveRoot}/Epoch_{epoch}gt.jpg' )
        for pre, gt in zip(pres_a, gts_a):
            num += 1
            print('ps',pre.shape, gt.shape)
            concatImg = np.concatenate([pre[0], divide_line, gt[0]], axis=1)
            #fromarray(cv2.cvtColor(raw_mask, cv2.COLOR_BGR2RGB))
            concatImg = Image.fromarray(cv2.cvtColor(concatImg.transpose(1, 2, 0), cv2.COLOR_BGR2RGB))#Image.fromarray(concatImg.transpose(1, 2, 0)).convert('RGB')
            concatImg.save(f'{saveRoot}/Epoch_{epoch}valid_{num}.jpg' )

        print('*********************************************************')
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()
        print('Eval Epoch: %s, PSNR: %s, SSIM: %s, Best_PSNR: %s, Best_SSIM: %s'
            %(epoch, psnr, ssim, best_psnr, best_ssim))



if __name__ == '__main__':
    ##########################################################################################################
    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(cfg['loader']['batch_size'], timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(cfg['loader']['batch_size'], timestamp2, args.save_name)

    save_root = osp.join(args.save_dir, timestamp1)
    save_path = osp.join(save_root, save_folder_name)



    # show configurations
    cfg_str = pprint.pformat(cfg)


    train_writer = SummaryWriter(save_path)

    ## Create model

    metrics = Metrics()
# increased number of dimensions will cause serious troubles? verify if it's enc or dec
    model = SpikeFormer(
        inputDim = 20,#spikeLen,
        dims = (16,32,64 ),      # dimensions of each stage 160-> 128
        heads = (1,2,5 ),           # heads of each stage
        ff_expansion = (8, 8, 4 ),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2 ), # reduction ratio of each stage for efficient attention
        num_layers = 1,                 # num layers of each stage o 2
        decoder_dim = 8,              # decoder dimension o 256
        out_channel = 1,                 # channel of restored image
        idim = 8
    )


    
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        #_log.info('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model)
        model.load_state_dict(network_data)
    else:
        network_data = None
        print('=> train flow model from scratch')
        #model.init_weights()
        print('=> Flow model params: {:.6f}M'.format(model.num_parameters()/1e6))
        if torch.cuda.device_count()>0:
            model = torch.nn.DataParallel(model).cuda()#.to(cpu)
        else:
            model = model.cuda()

    cudnn.benchmark = True
    print(model)

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']

    print('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.module.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]

    optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))

    ## Dataset: CFG AND H5 LOAD MINIZED  
    train_set = H5Loader_rssf_train(cfg)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last = True,
        batch_size = cfg['loader']['batch_size'],
        shuffle = False, #
        # pin_memory = False,
        # prefetch_factor = 6,
        num_workers = args.num_workers)
    
    epoch = args.start_epoch
    saveRoot = f'ckpt_{timestamp2}'
    os.mkdir(saveRoot)
  
    if args.eval:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        validation(epoch,cfg=cfg, test_datasets=test_datasets, model=model)
    else:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        
        while(True):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,

                train_writer=train_writer)
            epoch += 1

            if epoch % args.decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.decay_factor


            # validation stage 

            
            if epoch % 11 == 1:
                print('validation commence')   
                validation(
                        epoch,cfg=cfg, 
                        test_datasets=test_datasets, 
                        model=model, 
                        saveRoot=saveRoot)

            print('Save Model')
            flow_model_save_name = '{:s}_epoch{:03d}.pth'.format(cfg['model']['flow_arch'], epoch)
            torch.save(model.state_dict(), osp.join(save_path, flow_model_save_name))

            if epoch >= cfg['loader']['n_epochs']:
                break