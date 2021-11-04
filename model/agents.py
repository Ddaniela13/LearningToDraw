import collections
import os
from collections import OrderedDict

import torchvision.models as models
from torch.nn import Flatten

import torchbearer as tb
# noinspection PyUnresolvedReferences
from dsketch.experiments.characters.models.encoders import *
from dsketch.experiments.characters.models.model_bases import _Base
# noinspection PyUnresolvedReferences
from dsketch.experiments.characters.models.recurrent_decoders import *
# noinspection PyUnresolvedReferences
from dsketch.experiments.characters.models.single_pass_decoders import *

MLP_IMG_FEAT = tb.state_key('mlp_im_feat')
MLP_SK_FEAT = tb.state_key('mlp_sk_feat')


class _RxBase(_Base, ABC):
    @classmethod
    def add_args(cls, p):
        cls._add_args(p)
        p.add_argument("--rx-latent-size", help="size of reciever latent space", type=int, default=64, required=False)


class SimpleReceiverMLP(_RxBase):
    """
    Receiver with mlp with one hidden layer
    """

    def __init__(self, input_sz=64, hidden=128, latent=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_sz, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent))

    @staticmethod
    def _add_args(p):
        p.add_argument("--rx-hidden", help="receiver mlp hidden size", type=int, default=64, required=False)

    @staticmethod
    def create(args):
        return SimpleReceiverMLP(input_sz=args.latent_size, hidden=args.rx_hidden, latent=args.rx_latent_size)

    def forward(self, inp, state=None):
        return self.mlp(inp)



def createVGG16(pretrained=False, sin_pretrained=False, dev='cuda:0'):
    assert not (pretrained and sin_pretrained), "Select imagenet-weights or sin-weights, not both!"

    if sin_pretrained:
        print("SIN weights enabled")
        # download model from URL manually and save to desired location
        filepath = "/ssd//vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth"

        assert os.path.exists(
            filepath), "Please download the VGG model yourself from the following link and save it locally: " \
                       "https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be " \
                       "downloaded automatically like the other models)"

        vgg16 = models.vgg16(pretrained=False)
        checkpoint = torch.load(filepath, map_location=dev)
        state_dict = checkpoint['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace(".module", '') # removing ‘.moldule’ from key] # remove module.
            new_state_dict[name] = v
        
        vgg16.load_state_dict(new_state_dict)
    else:
        vgg16 = models.vgg16(pretrained=pretrained)
    return vgg16


# encoder for coloured (3channel input) CIFAR10 images
class VGG16Backbone(Encoder):
    def __init__(self, latent=64, pretrained=False, sin_pretrained=False, freeze_vgg=False, dev='cuda:0'):
        super().__init__(3)

        print(dev)
        vgg16 = createVGG16(pretrained, sin_pretrained, dev)
        self.vgg16 = vgg16.features[:16]  # 256 feature output
        if freeze_vgg:
            for p in self.vgg16.parameters():
                p.requires_grad = False

        self.enc = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            Flatten(),
            nn.Linear(8 * 8 * 256, latent)
        )

    @staticmethod
    def _add_args(p):
        p.add_argument("--imagenet-weights", help="Use imagenet pretrained weights", action='store_true',
                       required=False)
        p.add_argument("--sin-weights", help="Use imagenet pretrained weights", action='store_true',
                       required=False)
        p.add_argument("--freeze-vgg", help="Freeze the vgg16 part of the weights", action='store_true', required=False)

    @staticmethod
    def create(args):
        return VGG16Backbone(latent=args.latent_size, pretrained=args.imagenet_weights, sin_pretrained=args.sin_weights,
                             freeze_vgg=args.freeze_vgg)

    def forward(self, x, state=None):
        x = self.vgg16(x)
        x = self.enc(x)

        return x


FEATURE_MAPS = tb.state_key('feature_maps')

VGG16_LAYER_KEYS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
VGG16_LAYERS = {'relu1_2': (4, 64, 1),
                'relu2_2': (9, 128, 2),
                'relu3_3': (16, 256, 4),
                'relu4_3': (23, 512, 8),
                'relu5_3': (30, 512, 16)}


class VGG16BackboneExtended(Encoder):
    def __init__(self, size, latent=64, layer='relu3_3', pretrained=False, sin_pretrained=False, freeze_vgg=False, dev='cuda:0'):
        super().__init__(3)

        vgg16 = createVGG16(pretrained, sin_pretrained, dev)

        layidx, fms, ds = VGG16_LAYERS[layer]
        shape = size // ds

        self.vgg16 = vgg16.features[:layidx]
        if freeze_vgg:
            for p in self.vgg16.parameters():
                p.requires_grad = False

        self.enc = nn.Sequential(
            Flatten(),
            nn.Linear(shape * shape * fms, latent)
        )

        self.fstate = None
        for i in range(len(VGG16_LAYER_KEYS)):
            key = VGG16_LAYER_KEYS[i]
            lay = VGG16_LAYERS[key][0]
            self.vgg16[lay - 1].register_forward_hook(self.make_hook(key))
            if key == layer:
                break
        self.enc.register_forward_hook(self.make_hook("projection"))

    @staticmethod
    def _add_args(p):
        p.add_argument("--imagenet-weights", help="Use imagenet pretrained weights", action='store_true',
                       required=False)
        p.add_argument("--sin-weights", help="Use imagenet pretrained weights", action='store_true',
                       required=False)
        p.add_argument("--feature-layer", help="Layer from which to extract features", choices=VGG16_LAYER_KEYS,
                       required=False, default='relu3_3')
        p.add_argument("--freeze-vgg", help="Freeze the vgg16 part of the weights", action='store_true', required=False)

    def make_hook(self, name):
        def hook(module, input, output):
            self.fstate[name] = output

        return hook

    @staticmethod
    def create(args):
        return VGG16BackboneExtended(size=args.size, latent=args.latent_size, layer=args.feature_layer,
                                     pretrained=args.imagenet_weights, sin_pretrained=args.sin_weights,
                                     freeze_vgg=args.freeze_vgg, dev=args.device)

    def forward(self, x, state=None):
        self.fstate = collections.OrderedDict()

        x = self.vgg16(x)
        x = self.enc(x)

        if state is not None:
            state[FEATURE_MAPS] = self.fstate

        return x

    
# RESNET50_LAYER_KEYS = ['layer1', 'layer2', 'layer3', 'layer4']
# RESNET50_LAYERS = {'layer1': (4, 256, 1),
#                 'layer2': (5, 512, 2),
#                 'layer3': (6, 1024, 4),
#                 'layer4': (7, 2048, 8)}

    
class ResNet50Backbone(Encoder):
    def __init__(self, size, latent=64, pretrained=False, freeze_resnet50=False, dev='cuda:0'):
        super().__init__(3)

        resnet50 = models.resnet50(pretrained=pretrained)

        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])# my_resnet50(layer, pretrained) #-1 if I want all resnet-50 blocks
        
        if freeze_resnet50:
            for p in self.resnet50.parameters():
                p.requires_grad = False

        self.enc = nn.Sequential(
            Flatten(),
            nn.Linear(2048, latent)#layer 7 cu 2048, 6 cu 1024, 5 cu 512, 4 cu 256
        )

        self.fstate = None
        self.resnet50[7].register_forward_hook(self.make_hook("layer7"))
        
        self.enc.register_forward_hook(self.make_hook("projection"))

    @staticmethod
    def _add_args(p):
        p.add_argument("--imagenet-weights", help="Use imagenet pretrained weights", action='store_true',
                       required=False)
        p.add_argument("--freeze-resnet50", help="Freeze the resnet50 part of the weights", action='store_true', required=False)

    def make_hook(self, name):
        def hook(module, input, output):
            self.fstate[name] = output

        return hook

    @staticmethod
    def create(args):
        return ResNet50Backbone(size=args.size, latent=args.latent_size,
                                     pretrained=args.imagenet_weights,
                                     freeze_resnet50=args.freeze_resnet50, dev=args.device)

    def forward(self, x, state=None):
        self.fstate = collections.OrderedDict()

        x = self.resnet50(x)
        x = self.enc(x)
        
        if state is not None:
            state[FEATURE_MAPS] = self.fstate

        return x

    
import clip

CLIP_MODEL_BASES = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
TRANFORMER_RESBLOCKS_KEYS = ['resblock0', 'resblock1', 'resblock2', 'resblock3', 'resblock4','resblock5', 'resblock6', 'resblock7', 'resblock8', 'resblock9','resblock10', 'resblock11']
CLIP_TRANFORMER_RESBLOCKS = {'resblock0': 0,
                'resblock1': 1,
                'resblock2': 2,
                'resblock3': 3,
                'resblock4': 4,
                'resblock5': 5,
                'resblock6': 6,       
                'resblock7': 7,
                'resblock8': 8,            
                'resblock9': 9,             
                'resblock10': 10,             
                'resblock11': 11}

# CLIP_TRANFORMER_RESBLOCKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


class CLIPVisualEncoder(Encoder):
    def __init__(self, size, latent=64, clip_model='ViT-B/32', freeze_model=False, transformer_resblock='resblock11', dev='cuda:0'):
        super().__init__(3)
        
        self.clipmodel, preprocess = clip.load(clip_model, device=dev)

        if freeze_model:
            for p in self.clipmodel.parameters():
                p.requires_grad = False

        res = self.clipmodel.visual.input_resolution
        dim = self.clipmodel.encode_image(torch.zeros(1, 3, res, res).to(dev)).shape[1]

        self.enc = nn.Sequential(
            Flatten(),
            nn.Linear(dim, latent) #hardcoding
        )


        self.fstate = None
        for i in range(len(TRANFORMER_RESBLOCKS_KEYS)):
            key = TRANFORMER_RESBLOCKS_KEYS[i]
            lay = CLIP_TRANFORMER_RESBLOCKS[key]
            self.clipmodel.visual.transformer.resblocks[lay].register_forward_hook(self.make_hook(key))
            if key == transformer_resblock:
                break
        self.enc.register_forward_hook(self.make_hook("projection"))
        self.clipmodel.visual.register_forward_hook(self.make_hook("CLIPvisual"))


    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape)==3:
                self.fstate[name] = output.permute(1, 0, 2) #LND -> NLD bs, smth, 768
            else:
                self.fstate[name] = output

        return hook    

    @staticmethod
    def _add_args(p):
        p.add_argument("--clip-model", help="Model base for CLIP visual encoder", choices=CLIP_MODEL_BASES,
                       required=False, default='ViT-B/32')
        p.add_argument("--freeze-model", help="Freeze the CLIP visual encoder part of the weights", action='store_true', required=False)
        p.add_argument("--transformer-resblock", help="Residual block in transformer part of ViT Clip model", choices=CLIP_TRANFORMER_RESBLOCKS, required=False, default='resblock11')

    
    @staticmethod
    def create(args):
        return CLIPVisualEncoder(size=args.size, latent=args.latent_size, clip_model=args.clip_model,
                                     freeze_model=args.freeze_model, dev=args.device)

    def forward(self, x, state=None):
        self.fstate = collections.OrderedDict()

        image_features = self.clipmodel.encode_image(x).float()
        out = self.enc(image_features)
        if state is not None:
            state[FEATURE_MAPS] = self.fstate
            
        return out
