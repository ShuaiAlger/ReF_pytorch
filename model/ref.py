import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn

class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)
      
      
class ReF(BaseNet):
    def __init__(self,  **kw):
        super(ReF, self).__init__()
    
        # num is 1 because of gray scale image
        self.input_channels = 3
        self.group_size = 4

        def e2conv(in_type, out_type, ksize, stride, dilation, padding, relu=True):
            return e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, bias=False), 
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=relu)
        )

        self.r2_act = e2cnn.gspaces.Rot2dOnR2(N=self.group_size)

        def trivial_type(c):
            r2_act = self.r2_act
            return e2cnn.nn.FieldType(r2_act, c * [r2_act.trivial_repr])
        def regular_type(c):
            r2_act = self.r2_act
            return e2cnn.nn.FieldType(r2_act, c * [r2_act.regular_repr])

        self.ops = e2cnn.nn.ModuleList([])

        # definition of rotation-equivariant block
        self.enc_list = [32//self.group_size,
                         64//self.group_size,
                         128//self.group_size,
                         256//self.group_size,
                         512//self.group_size]
        
        self.ops.append(e2conv(trivial_type(self.input_channels), regular_type(self.enc_list[0]), 3, 1, 1, 1))
        self.ops.append(e2conv(regular_type(self.enc_list[0]), regular_type(self.enc_list[1]), 3, 1, 1, 1))
        self.ops.append(e2conv(regular_type(self.enc_list[1]), regular_type(self.enc_list[2]), 3, 1, 2, 2))
        self.ops.append(e2conv(regular_type(self.enc_list[2]), regular_type(self.enc_list[3]), 3, 1, 2, 2))
        self.ops.append(e2conv(regular_type(self.enc_list[3]), regular_type(self.enc_list[4]), 3, 1, 1, 1))
                

        self.decoder1 = torch.nn.Sequential(torch.nn.Conv2d(self.enc_list[-1], 2, 1))
        self.decoder2 = torch.nn.Sequential(torch.nn.Conv2d(self.enc_list[-1], 2, 1))


    def forward_one(self, x):
        r2_act = self.r2_act
        x = e2cnn.nn.GeometricTensor(x, e2cnn.nn.FieldType(r2_act,
                                     self.input_channels * [r2_act.trivial_repr]))
        for op in self.ops:
            x = op(x)

        codes = e2cnn.nn.GroupPooling(e2cnn.nn.FieldType(r2_act, self.enc_list[-1] * [r2_act.regular_repr]))(x).tensor
        descriptors = codes
        codes = torch.square(codes)

        ureliability = self.decoder1(codes)
        urepeatability = self.decoder2(codes)
        return self.normalize(descriptors, ureliability, urepeatability)
