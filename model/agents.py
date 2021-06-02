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
                       required=False, default='relu5_3')
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
