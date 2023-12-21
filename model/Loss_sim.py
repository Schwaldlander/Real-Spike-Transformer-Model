import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        #print('diff', x.shape, y.shape)# 1: diff torch.Size([1, 1, 320, 448]) torch.Size([1, 3600, 1280, 3])
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss.requires_grad_(True)


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        # self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).unsqueeze(0).repeat(1,1,1,1).cpu() #这个的repeat也是后加的
        #MODI FIED FROM 3 TO 6 
        # print(self.kernel.shape)
        # if torch.cuda.is_available():
        #     self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        #print('aaaa image shape')
        #print(img.shape)
        #print('kernel',self.kernel.shape) # kernel torch.Size([1, 3, 5, 5])
        _,n_channels, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel)
        #return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        #print('filter', filtered.shape)
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        #print('nwf shape',new_filter.shape)
        filtered    = self.conv_gauss(new_filter.repeat(1,1,1,1)) # filter  #这里为什么需要repeat一下？原文的目的是什么？否则不能正常运行
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # print('x y',x.shape)
        # print(y.shape)
        _,_,k1,k2= y.shape
# x y torch.Size([1, 1, 320, 448])
#torch.Size([1, 1, 2, 320, 448])

        #y = y.repeat(1,2,1) repeat(1,2,1,1)
        x = x.repeat(1,1,1,1).cpu()
        y = y.cpu()
        #print('bbbbbb')
        # torch.Size([1, 18, 320, 448])
# torch.Size([1, 3, 320, 448])
        # print(x.shape)
        # print(y.shape)
        lapy = self.laplacian_kernel(y)
        loss = self.loss(self.laplacian_kernel(x), lapy)
        return loss.requires_grad_(True)


class VGGLoss4(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), #
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(inplace=True),
        )
        self.load_state_dict(torch.load(path))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, real_y, fake_y):
        #real_y.squeeze(0)#.view(1,0)
        #print('f r', real_y.shape, fake_y.shape)
        # f r torch.Size([1, 1, 320, 448]) torch.Size([3, 1, 2, 320, 448])

        #real_y = real_y.reshape(int((a*b)/3),3,c,d)
        # real_y = real_y.repeat((3, 1, 1, 1)).permute(1,0,2,3)
        # fake_y = fake_y.repeat((1, 3, 1, 1))
        _,_,k1,k2= fake_y.shape
        real_y = real_y.repeat((1,3,1,1)).cpu()
        fake_y = fake_y.repeat((1,3,1,1)).cpu()
        # fake_y = fake_y.reshape(2,3,k1,k2)
        with torch.no_grad():
            real_f = self.features(real_y)
        fake_f = self.features(fake_y)
        #print('FF', real_f.shape, fake_f.shape)
        msel = F.mse_loss(real_f, fake_f)
        return msel.requires_grad_(True)

    
