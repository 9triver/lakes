"""Paper-structure adaptations; implementation decisions are documented in docs/paper_model_adaptations.md."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, dropout=.3):
        super().__init__()
        self.project=nn.Conv2d(incoming,outgoing,1)
        self.body=nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing),nn.ReLU(),nn.Dropout(dropout),
            nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),nn.BatchNorm2d(outgoing),nn.ReLU())

    def forward(self,x):
        return self.body(x)+self.project(x)


class AttentionGate(nn.Module):
    def __init__(self,channels):
        super().__init__()
        hidden=max(1,channels//2)
        self.skip=nn.Conv2d(channels,hidden,1)
        self.gate=nn.Conv2d(channels,hidden,1)
        self.score=nn.Conv2d(hidden,1,1)

    def forward(self,skip,gate):
        return skip*torch.sigmoid(self.score(F.relu(self.skip(skip)+self.gate(gate))))


class AERUNetAdapted(nn.Module):
    def __init__(self,in_channels,base_channels=32):
        super().__init__()
        widths=[base_channels*2**i for i in range(4)]
        self.encoders=nn.ModuleList(ResidualBlock(in_channels if i==0 else widths[i-1],widths[i]) for i in range(3))
        self.pool=nn.MaxPool2d(2)
        self.center=nn.Sequential(ResidualBlock(widths[2],widths[3]),nn.Dropout(.3))
        self.ups=nn.ModuleList(nn.ConvTranspose2d(widths[i+1],widths[i],2,stride=2) for i in [2,1,0])
        self.gates=nn.ModuleList(AttentionGate(widths[i]) for i in [2,1,0])
        self.decoders=nn.ModuleList(ResidualBlock(widths[i]*2,widths[i]) for i in [2,1,0])
        self.head=nn.Conv2d(widths[0],1,1)

    def forward(self,x):
        skips=[]
        for encoder in self.encoders:
            x=encoder(x);skips.append(x);x=self.pool(x)
        x=self.center(x)
        for up,gate,decoder,skip in zip(self.ups,self.gates,self.decoders,reversed(skips),strict=True):
            x=up(x)
            x=decoder(torch.cat([x,gate(skip,x)],dim=1))
        return self.head(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.LayerNorm(channels,eps=1e-6)

    def forward(self,x):
        return self.norm(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()


class GlobalResponseNorm(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.gamma=nn.Parameter(torch.zeros(1,channels,1,1))
        self.beta=nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self,x):
        magnitude=torch.linalg.vector_norm(x,dim=(2,3),keepdim=True)
        relative=magnitude/(magnitude.mean(dim=1,keepdim=True)+1e-6)
        return x+self.gamma*(x*relative)+self.beta


class ConvNeXtV2Block(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(channels,channels,7,padding=3,groups=channels),
            ChannelLayerNorm(channels),nn.Conv2d(channels,channels*4,1),nn.GELU(),
            GlobalResponseNorm(channels*4),nn.Conv2d(channels*4,channels,1))

    def forward(self,x):return x+self.body(x)


def gaussian_modulation(x):
    # Per-image statistics make tiled inference independent of the other images in a batch.
    axes=tuple(range(1,x.ndim))
    mean=x.mean(dim=axes,keepdim=True).detach()
    std=x.std(dim=axes,keepdim=True,unbiased=False).detach().clamp_min(1e-4)
    return torch.exp(-.5*((x-mean)/std).square())/(std*math.sqrt(2*math.pi))


class GaussianAMM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        hidden=max(1,channels//16)
        self.channel=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,hidden,1),
            nn.ReLU(),nn.Conv2d(hidden,channels,1))
        self.spatial=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),
            nn.BatchNorm2d(1,momentum=.01))

    def forward(self,x):
        x=x*gaussian_modulation(self.channel(x))
        return x*gaussian_modulation(self.spatial(torch.cat([x.amax(dim=1,keepdim=True),x.mean(dim=1,keepdim=True)],dim=1)))


class CAStage(nn.Module):
    def __init__(self,channels):
        super().__init__();self.conv=ConvNeXtV2Block(channels);self.amm=GaussianAMM(channels)

    def forward(self,x):
        x=self.conv(x);return x+self.amm(x)


class ResDoubleDSC(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__()
        self.shortcut=nn.Conv2d(incoming,outgoing,1,bias=False)
        self.body=nn.Sequential(nn.Conv2d(incoming,incoming,3,padding=1,groups=incoming,bias=False),
            nn.Conv2d(incoming,outgoing,1,bias=False),nn.BatchNorm2d(outgoing),nn.ReLU(),
            nn.Conv2d(outgoing,outgoing,3,padding=1,groups=outgoing,bias=False),
            nn.Conv2d(outgoing,outgoing,1,bias=False),nn.BatchNorm2d(outgoing))

    def forward(self,x):return F.relu(self.shortcut(x)+self.body(x))


class ResDDSCUNetPPAdapted(nn.Module):
    def __init__(self,in_channels,base_channels=32):
        super().__init__()
        widths=[base_channels*2**i for i in range(5)]
        self.stem=nn.Conv2d(in_channels,widths[0],3,padding=1)
        self.stages=nn.ModuleList(CAStage(c) for c in widths)
        self.down=nn.ModuleList(nn.Sequential(ChannelLayerNorm(widths[i]),nn.Conv2d(widths[i],widths[i+1],2,stride=2)) for i in range(4))
        self.up=nn.ModuleDict();self.decoder=nn.ModuleDict()
        for j in range(1,5):
            for i in range(5-j):
                key=f'{i}_{j}'
                self.up[key]=nn.Conv2d(widths[i+1],widths[i],1)
                self.decoder[key]=ResDoubleDSC((j+1)*widths[i],widths[i])
        self.heads=nn.ModuleList(nn.Conv2d(widths[0],1,1) for _ in range(4))

    def forward(self,x):
        nodes={};x=self.stem(x)
        for i,stage in enumerate(self.stages):
            if i:x=self.down[i-1](x)
            x=stage(x);nodes[i,0]=x
        for j in range(1,5):
            for i in range(5-j):
                key=f'{i}_{j}'
                up=F.interpolate(self.up[key](nodes[i+1,j-1]),size=nodes[i,0].shape[-2:],mode='bilinear',align_corners=False)
                nodes[i,j]=self.decoder[key](torch.cat([*[nodes[i,k] for k in range(j)],up],dim=1))
        return torch.stack([head(nodes[0,j]) for j,head in enumerate(self.heads,1)]).mean(dim=0)
