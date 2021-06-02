import torch
import torch.nn as nn

import torchbearer
from utils.shared_datasets import IMAGENET_NORM

SKETCHES = torchbearer.state_key('sketches')
SENDER_IMAGES = torchbearer.state_key('sender_images')
RECEIVER_IMAGES = torchbearer.state_key('receiver_images')
RECEIVER_IMAGES_MATCHED = torchbearer.state_key('receiver_images_matched_order')


class SketchingGame(nn.Module):
    def __init__(self, sender_encoder, receiver_encoder, decoder, receiver, num_targets_per_game, imagenet_norm=False,
                 invert=False):
        super().__init__()

        self.sender_encoder = sender_encoder
        self.receiver_encoder = receiver_encoder
        self.decoder = decoder
        self.receiver = receiver
        self.imagenet_norm = imagenet_norm
        self.invert = invert
        self.num_targets_per_game = num_targets_per_game

    def get_feature(self, x):
        return self.encoder(x, None)

    def forward(self, sender_x, receiver_x, state=None):
        if self.receiver_encoder.channels == 3 and sender_x.shape[1] == 1:
            sender_x = torch.cat(3 * [sender_x], dim=1)  # to then pass through the vgg16 FE

        sender_im_features = self.sender_encoder(sender_x, state)
        sketches = self.decoder(sender_im_features)

        if self.invert:
            sketches = 1 - sketches

        if state is not None:
            state[SKETCHES] = sketches

        if self.receiver_encoder.channels == 3:
            sketches = torch.cat(3 * [sketches], dim=1)  # to then pass through the vgg16 FE

        if self.imagenet_norm:
            sketches = IMAGENET_NORM(sketches)  # normalise like imagenet if required

        rx_im_features = self.receiver(self.receiver_encoder(receiver_x))
        rx_sk_features = self.receiver(self.receiver_encoder(sketches))

        if self.num_targets_per_game is None:
            return (rx_im_features @ rx_sk_features.t()).t()
            
        scores = []
        for i in range(0, sender_x.shape[0], self.num_targets_per_game):
            _rx_im_features = rx_im_features[i: i + self.num_targets_per_game]
            _rx_sk_features = rx_sk_features[i: i + self.num_targets_per_game]
            fr = (_rx_im_features @ _rx_sk_features.t()).t()
            scores.append(fr)

        return torch.cat(scores, dim=0)

    def get_callbacks(self, args):
        if self.sender_encoder == self.receiver_encoder:
            return [*self.sender_encoder.get_callbacks(args), *self.decoder.get_callbacks(args),
                    *self.receiver.get_callbacks(args)]
        else:
            return [*self.sender_encoder.get_callbacks(args), *self.receiver_encoder.get_callbacks(args),
                    *self.decoder.get_callbacks(args), *self.receiver.get_callbacks(args)]

