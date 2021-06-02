import argparse
from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

import torchbearer
from utils.game_data import GameSampler, build_dataloaders, build_game_loader, pair_images


class TestGameSampler(TestCase):
    def test_classes_correct_noshuffle(self):
        ds = FakeData(num_classes=10, image_size=(1, 8, 8), transform=ToTensor())
        gs = GameSampler(ds, shuffle=False)
        dl = DataLoader(ds, batch_sampler=gs, num_workers=4)

        for img, tgt in dl:
            self.assertEqual(len(tgt), len(set(tgt)))

    def test_classes_correct_shuffle(self):
        ds = FakeData(num_classes=10, image_size=(3, 32, 32))
        gs = GameSampler(ds, shuffle=True)
        dl = DataLoader(ds, batch_sampler=gs, num_workers=4)

        for img, tgt in dl:
            self.assertEqual(len(tgt), len(set(tgt)))


class TestBuildDataloaders(TestCase):
    def test_oo_game3_1(self):
        args = argparse.Namespace()
        args.num_classes = 101
        args.dataset = "_Testing"
        args.image_size = 2
        args.data_seed = 0
        args.batch_size = 64
        args.num_workers = 2
        args.additional_transforms = pair_images

        args.sender_images_per_iter = None
        args.num_targets = None
        args.object_oriented = "different"

        self._oo_game3_tests(args)

    def test_oo_game3_2(self):
        args = argparse.Namespace()
        args.num_classes = 101
        args.dataset = "_Testing"
        args.image_size = 2
        args.data_seed = 0
        args.batch_size = 202
        args.num_workers = 2
        args.additional_transforms = pair_images

        args.sender_images_per_iter = None
        args.num_targets = None
        args.object_oriented = "different"

        self._oo_game3_tests(args)

    def test_oo_game3_3(self):
        args = argparse.Namespace()
        args.num_classes = 101
        args.dataset = "_Testing"
        args.image_size = 2
        args.data_seed = 0
        args.batch_size = 303
        args.num_workers = 2
        args.additional_transforms = pair_images

        args.sender_images_per_iter = None
        args.num_targets = None
        args.object_oriented = "different"

        self._oo_game3_tests(args)

    def _oo_game3_tests(self, args):
        trainloader, valloader, testloader = build_dataloaders(args)
        self.assertEqual(args.num_targets, args.num_classes)

        loader = build_game_loader(args)

        state = {
            torchbearer.ITERATOR: iter(trainloader),
            torchbearer.DEVICE: "cpu",
            torchbearer.DATA_TYPE: torch.float32
        }
        for _ in range(len(trainloader)):
            loader(state)

            all_sender_images, all_target_images = state[torchbearer.X]
            all_targets = state[torchbearer.Y_TRUE]

            for j in range(0, all_sender_images.shape[0], args.num_targets):
                sender_images = all_sender_images[j: j + args.num_targets]
                target_images = all_target_images[j: j + args.num_targets]
                targets = all_targets[j: j + args.num_targets] - j

                for i, tgt in enumerate(targets.tolist()):
                    self.assertTrue(torch.allclose(sender_images[i][0], target_images[tgt][0]))
                    self.assertEqual(sender_images[i].shape[0], 3)
                    self.assertEqual(sender_images[i].shape[1], 2)
                    self.assertEqual(sender_images[i].shape[2], 2)

                inplabels = set(sender_images[:, 0, 0, 0].tolist())
                tgtlabels = set(target_images[:, 0, 0, 0].tolist())
                inpind = set(sender_images[:, 1, 0, 0].tolist())
                tgtind = set(target_images[:, 1, 0, 0].tolist())

                self.assertEqual(len(inpind.union(tgtind)),
                                 args.num_classes * 2)  # not expecting any of the targets to be in the senders
                self.assertEqual(len(inpind), args.num_classes)  # all classes
                self.assertEqual(len(tgtind), args.num_classes)  # all classes
                self.assertEqual(len(inplabels.difference(tgtlabels)), 0)  # expecting all classes both sides
