import sys

import torch.nn.functional as F
from torch import nn

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

    return in_tens.mean([2, 3], keepdim=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def make_perceptual_loss(weights):
    @add_to_loss
    def perceptual_loss(state):
        outs0 = state[FEATURE_MAPS]

        game = state[tb.MODEL]
        sketches = state[SKETCHES]
        sketches = torch.cat(3 * [sketches], dim=1)
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

        return val.mean()

    return perceptual_loss


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
        sketches = torch.cat(3 * [sketches], dim=1)
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
        sketches = torch.cat(3 * [sketches], dim=1)
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
