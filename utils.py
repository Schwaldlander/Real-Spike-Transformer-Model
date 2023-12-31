import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import bilinear_sampler, coords_grid, upflow8
from model.corr import CorrBlock
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)
    return

class WarmUp():
    def __init__(self, ed_it, st_lr, ed_lr):
        self.ed_it = ed_it
        self.st_lr = st_lr
        self.ed_lr = ed_lr
    
    def get_lr(self, cur_it):
        return self.st_lr + (self.ed_lr - self.st_lr) / self.ed_it * cur_it

    def adjust_lr(self, optimizer, cur_it):
        if cur_it <= self.ed_it:
            lr = self.get_lr(cur_it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

class InputPadder:
    """ Pads images such that dimensions are divisible by 16 """
    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 16) + 1) * 16 - self.ht) % 16
        pad_wd = (((self.wd // 16) + 1) * 16 - self.wd) % 16
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def evaward(self, spks, iters=8, flow_init=None, upsample=True, test_mode=False, phm=False):
        """ Estimate optical flow between pair of frames """

        if phm:
            spk0 = spks[:, 0:21].contiguous()
            spk1 = spks[:, 10:31].contiguous()
            spk2 = spks[:, 20:41].contiguous()
            spk3 = spks[:, 30:51].contiguous()
        else:
            spk0 = spks[:, 10:31].contiguous()
            spk1 = spks[:, 30:51].contiguous()
            spk2 = spks[:, 50:71].contiguous()
            spk3 = spks[:, 70:91].contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap0, fmap1, fmap2, fmap3 = self.fnet([spk0, spk1, spk2, spk3])        
        
        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fmap3 = fmap3.float()
        
        corr_fn1 = CorrBlock(fmap0, fmap1, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        corr_fn2 = CorrBlock(fmap0, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        corr_fn3 = CorrBlock(fmap0, fmap3, num_levels=self.args.corr_levels, radius=self.args.corr_radius)


        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(spk0)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1, coords2, coords3 = self.initialize_flow(spk0)

        if flow_init is not None:
            coords1 = coords1 + flow_init[0]
            coords2 = coords2 + flow_init[1]
            coords3 = coords3 + flow_init[2]

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            coords2 = coords2.detach()
            coords3 = coords3.detach()
            
            corr1 = corr_fn1(coords1) # index correlation volume
            corr2 = corr_fn2(coords2) # index correlation volume
            corr3 = corr_fn3(coords3) # index correlation volume

            flow1 = coords1 - coords0
            flow2 = coords2 - coords0
            flow3 = coords3 - coords0

            corr_list = [corr1, corr2, corr3]
            flow_list = [flow1, flow2, flow3]
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow_list = self.update_block(net, inp, corr_list, flow_list)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow_list[0]
            coords2 = coords2 + delta_flow_list[1] * 2
            coords3 = coords3 + delta_flow_list[2] * 3

            # upsample predictions
            if up_mask is None:
                flow_up = []
                flow_up.append(upflow8(coords1 - coords0))
                flow_up.append(upflow8(coords2 - coords0))
                flow_up.append(upflow8(coords3 - coords0))
            else:
                flow_up = []
                flow_up.append(self.upsample_flow(coords1 - coords0, up_mask))
                flow_up.append(self.upsample_flow(coords2 - coords0, up_mask))
                flow_up.append(self.upsample_flow(coords3 - coords0, up_mask))
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords3 - coords0, flow_up
            
        return flow_predictions

    

def supervised_loss(flow, flow_gt, gamma=0.8):
    #flow = flowm()

    n_predictions = len(flow)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        for j in range(len(flow_gt)):
            i_loss = (flow[i][j] - flow_gt[j]).abs()
            flow_loss += i_weight * i_loss.mean() / len(flow_gt)
    loss_deriv_dict = {}
    loss_deriv_dict['flow_mean'] = flow[-1][0].abs().mean()
    return flow_loss, loss_deriv_dict

##########################################################################################################
## Compute Error

def get_class_metric(metric_dict, len_dict):
    CLASS_A = [
        '2016-09-02_000170',
        'car_tuebingen_000024',
        'car_tuebingen_000145',
        'motocross_000108']
    CLASS_B = [
        'ball_000000',
        'kids__000002',
        'skatepark_000034',
        'spitzberglauf_000009']
    CLASS_C = [
        'car_tuebingen_000103',
        'horses__000028',
        'tubingen_05_09_000012']
    
    a_metric_sum = 0
    b_metric_sum = 0
    c_metric_sum = 0
    a_len_sum = 0
    b_len_sum = 0
    c_len_sum = 0
    for k, v in metric_dict.items():
        if k in CLASS_A:
            a_metric_sum += v * len_dict[k]
            a_len_sum += len_dict[k]
        elif k in CLASS_B:
            b_metric_sum += v * len_dict[k]
            b_len_sum += len_dict[k]
        elif k in CLASS_C:
            c_metric_sum += v * len_dict[k]
            c_len_sum += len_dict[k]
    
    if a_len_sum!=0:
        return a_metric_sum/a_len_sum, b_metric_sum/b_len_sum, c_metric_sum/c_len_sum
    else: 
        return  a_metric_sum, b_metric_sum, c_metric_sum


def calculate_error_rate(pred_flow, gt_flow):
    print('pred flow',pred_flow.shape)
    pred_flow = pred_flow.squeeze(dim=0).cpu().numpy()#.permute([1,2,0])
    gt_flow = gt_flow.squeeze(dim=0).cpu().numpy()

    epe_map = np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2))
    bad_pixels = np.logical_and(
        epe_map > 0.5,
        epe_map / np.maximum(
            np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
    return bad_pixels.mean() * 100.



    
##########################################################################################################
## Flow Viz

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=True):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    print(flow_uv.shape)
    # assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    # assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def SaveModel(epoch, bestPerformance, model, optimizer, saveRoot, best=False):
    saveDict = {
        'pre_epoch':epoch,
        'performance':bestPerformance,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    savePath = os.path.join(saveRoot, '%s.pth' %('latest' if not best else 'best'))
    torch.save(saveDict, savePath)

def LoadModel(checkPath, model, optimizer=None):
    stateDict = torch.load(checkPath)
    pre_epoch = stateDict['pre_epoch']
    model.load_state_dict(stateDict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(stateDict['optimizer_state_dict'])

    return pre_epoch, stateDict['performance'], \
           stateDict['model_state_dict'], stateDict['optimizer_state_dict']
def vis_flow_batch(forw_flows_list, save_path, suffix='forw_flow', max_batch=4):
    # forward flow
    for ii in range(forw_flows_list[0].shape[0]):
        for jj in range(len(forw_flows_list)):
            forw_flows = forw_flows_list[jj]
            flow = forw_flows[ii, :]
            flow = flow.permute([1, 2, 0]).detach().cpu().numpy()
            if jj == 0:
                flow_viz = flow_to_image(flow, convert_to_bgr=True)
            else:
                flow_viz = np.concatenate([flow_viz, flow_to_image(flow, convert_to_bgr=True)], axis=1)        
        if ii == 0:
            flow_viz_all = flow_viz
        else:
            flow_viz_all = np.concatenate([flow_viz_all, flow_viz], axis=0)
        if ii-1 >= max_batch:
            break
    cur_save_path = osp.join(save_path, '{:s}.png'.format(suffix))
    cv2.imwrite(cur_save_path, flow_viz_all)

def vis_img_batch(imgs, save_path, max_batch=4):
    # img
    for ii in range(imgs.shape[0]):
        img = imgs[ii, :]
        img = img.permute([1, 2, 0]).detach().cpu().numpy() * 255.
        cur_save_path = osp.join(save_path, 'img_batch{:d}.png'.format(ii))
        cv2.imwrite(cur_save_path, img.astype(np.uint8))
        if ii-1 >= max_batch:
            break
