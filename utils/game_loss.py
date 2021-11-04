import sys

import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

import torchbearer
import torchbearer as tb
from dsketch.experiments.shared.args_losses import _Loss
from dsketch.experiments.shared.utils import list_class_names
from model import FEATURE_MAPS, SKETCHES, IMAGENET_NORM, torch
from torchbearer.callbacks import add_to_loss


class HingeLoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, logits, target, reduction='mean'):
        return F.multi_margin_loss(logits.cpu(), target.cpu(), margin=1, p=1, reduction=reduction)
    
    
class NoLoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, logits, target, reduction='mean'):
        return 0


@add_to_loss
def reconstruct_loss(state):
    sender_images = state['sender_images']
    sketches = state['sketches']
    return F.mse_loss(sketches, sender_images)


class CELoss(_Loss):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_args(p):
        pass

    def __call__(self, logits, target, reduction='mean'):
        return F.cross_entropy(logits, target, reduction=reduction)


def get_loss(name):
    los = getattr(sys.modules[__name__], name)
    if not issubclass(los, _Loss):
        raise TypeError()
    return los


def loss_choices():
    return list_class_names(_Loss, __name__)


def build_loss(args):
    base_loss = get_loss(args.loss)(args)

    if "games_per_batch" in args:
        return MultiGamesPerBatchLoss(base_loss, args.games_per_batch)

    return base_loss


def spatial_average(in_tens, keepdim=True):
    if len(in_tens.shape) == 2:
        in_tens = in_tens.view(in_tens.shape[0], 1, 1, 1)
        
    if len(in_tens.shape) == 3: #this is for ViT features which seem to have shape [bs, 50, 1, 768]
        in_tens = in_tens.view(in_tens.shape[0], in_tens.shape[1], 1, in_tens.shape[2])

    return in_tens.mean([2, 3], keepdim=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def make_perceptual_loss(weights, coef):
    @add_to_loss
    def perceptual_loss(state):
        outs0 = state[FEATURE_MAPS]

        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        
        if game.imagenet_norm:
            sketches = IMAGENET_NORM(sketches)  # normalise like imagenet if required
        game.sender_encoder(sketches)
        outs1 = game.sender_encoder.fstate
        
        feats0, feats1, diffs = {}, {}, {}
        for kk in outs0.keys():
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in outs0.keys()]

        val = weights[0] * res[0]
        for i in range(1, min(len(res), len(weights))):
            val += weights[i] * res[i]

        return coef * val.mean()

    return perceptual_loss


DOG_IMAGE = torchbearer.state_key("dog_image")


def make_dog_perceptual_loss(weights, coef, like_a_dog_image):
    # perceptual loss to make everything look like a dog
    @add_to_loss
    def perceptual_loss(state):
        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        if game.imagenet_norm:
            sketches = IMAGENET_NORM(sketches)  # normalise like imagenet if required
        game.sender_encoder(sketches)
        outs1 = game.sender_encoder.fstate

        if DOG_IMAGE not in state:
            print("Creating DOG_IMAGE features")
            with torch.no_grad():
                with Image.open(like_a_dog_image) as dogimg:
                    image_size = sketches.shape[-1]

                    img = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor()
                    ])(dogimg)
                    img = img.to(sketches.device)
                    img = img.unsqueeze(0)
                    if game.imagenet_norm:
                        img = IMAGENET_NORM(img)  # normalise like imagenet if required
                    game.sender_encoder(img)
                    state[DOG_IMAGE] = game.sender_encoder.fstate
        outs0 = state[DOG_IMAGE]

        feats0, feats1, diffs = {}, {}, {}
        for kk in outs0.keys():
            outs0kk = torch.cat(outs1[kk].shape[0] * [outs0[kk]], dim=0)
            feats0[kk], feats1[kk] = normalize_tensor(outs0kk), normalize_tensor(outs1[kk])

            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in outs0.keys()]

        val = weights[0] * res[0]
        for i in range(1, min(len(res), len(weights))):
            val += weights[i] * res[i]

        return coef * val.mean()

    return perceptual_loss


def make_CLIP_loss(args):
    @add_to_loss
    def CLIP_perceptual_loss(state):
        outs0 = state[FEATURE_MAPS]
        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        augment_trans = [
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(args.image_size, scale=(0.7, 0.9)),
        ]
        
        if game.imagenet_norm:
            augment_trans.append(IMAGENET_NORM)
        
        augment_trans=transforms.Compose(augment_trans)

        feats0, feats1, diffs = {}, {}, {}
    
        loss = 0
        NUM_AUGS = 4
        
        key='CLIPvisual'
        
        if key not in outs0.keys(): #in case the CLIPloss is used with vgg16 encoder, change feature layer key
            key='relu5_3'
            outs0[key]=outs0[key].flatten(1)
            
        for i in range(0, outs0[key].shape[0]):
            img_augs = []
            img_i_features = outs0[key][i].view(1,outs0[key].shape[1])
            
            sketch_i = sketches[i]
            
            for n in range(NUM_AUGS):
                img_augs.append(augment_trans(sketch_i))
            sketches_batch = torch.cat(img_augs).view(NUM_AUGS, 3, args.image_size, args.image_size)
            
            game.sender_encoder(sketches_batch)
            outs1=game.sender_encoder.fstate
            
            augbatch_sketches_features = outs1[key]
            if key=='relu5_3':
                augbatch_sketches_features=augbatch_sketches_features.flatten(1)
            
            for n in range(NUM_AUGS):
                  loss += 1 - torch.cosine_similarity(img_i_features, augbatch_sketches_features[n:n+1], dim=1) #compute cosine distance
#                 loss -= torch.cosine_similarity(img_i_features, augbatch_sketches_features[n:n+1], dim=1) #use cosine similarity

        return loss
        

    return CLIP_perceptual_loss


class MultiGamesPerBatchLoss:
    def __init__(self, loss, games_per_batch):
        self.loss = loss
        self.games_per_batch = games_per_batch

    def __call__(self, input, target):
        # input, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]
        losses = []
        gamesize = input.shape[0] // self.games_per_batch

        for i in range(0, input.shape[0], gamesize):
            inp = input[i: i + gamesize]
            tgt = target[i: i + gamesize] - i
            assert tgt.max() == gamesize - 1
            losses.append(self.loss(inp, tgt, reduction='none'))

        return torch.cat(losses, dim=0).mean()


class LearnablePerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(256, 1, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)
        self.convs = {
            'relu1_2': self.conv1,
            'relu2_2': self.conv2,
            'relu3_3': self.conv3,
            'relu4_3': self.conv4,
            'relu5_3': self.conv5
        }

    def forward(self, state):
        outs0 = state[FEATURE_MAPS]

        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        
        if game.imagenet_norm:
            sketches = IMAGENET_NORM(sketches)  # normalise like imagenet if required
        game.sender_encoder(sketches)
        outs1 = game.sender_encoder.fstate

        feats0, feats1, diffs = {}, {}, {}
        for kk in outs0.keys():
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.convs[kk](diffs[kk]), keepdim=True) for kk in outs0.keys()]

        val = res[0]
        for i in range(1, len(res)):
            val += res[i]

        return val.mean()


class LearnableWeightedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.rand(5))

    def forward(self, state):
        outs0 = state[FEATURE_MAPS]

        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        
        if game.imagenet_norm:
            sketches = IMAGENET_NORM(sketches)  # normalise like imagenet if required
        game.sender_encoder(sketches)
        outs1 = game.sender_encoder.fstate

        feats0, feats1, diffs = {}, {}, {}
        for kk in outs0.keys():
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in outs0.keys()]

        with torch.no_grad():
            self.weights.clamp_min_(0)
            self.weights.divide_(self.weights.norm())
        weights = self.weights * 1

        val = weights[0] * res[0]
        for i in range(1, len(res)):
            val += weights[i] * res[i]

        return val.mean()
