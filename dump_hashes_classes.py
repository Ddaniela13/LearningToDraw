import argparse
import hashlib

from PIL import Image
from torchvision.transforms import transforms

from dsketch.experiments.shared.utils import FakeArgumentParser
from utils.game_data import build_dataloaders
from utils.shared_datasets import dataset_choices, get_dataset, IMAGENET_NORM_INV


def add_subparsers(parser, add_help=False):
    parser.add_argument("--dataset", help="dataset", required=True, choices=dataset_choices())
    parser.add_argument("--data-seed", help='random seed for dataset shuffling, etc', required=False, type=int, default=1234)
    parser.add_argument("--name", help='print class name rather than id', required=False, action='store_true')

def add_sub_args(args, parser):
    if 'dataset' in args and args.dataset is not None:
        get_dataset(args.dataset).add_args(parser)


def main():
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_subparsers(fake_parser, add_help=False)
    fake_args, _ = fake_parser.parse_known_args()

    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_subparsers(parser)
    add_sub_args(fake_args, parser)

    args = parser.parse_args()
    args.object_oriented = None
    args.num_targets = 1
    args.sender_images_per_iter = 1


    inv = get_dataset(args.dataset).inv_transform

    args.imagenet_norm = True
    if 'imagenet_norm' in args and args.imagenet_norm:
        sinv = transforms.Compose([inv, IMAGENET_NORM_INV])
    else:
        sinv = inv

    if args.name:
        ds = get_dataset(args.dataset)
        _, _, test = ds.create(args)
        classes = test.classes

    traingen, valgen, testgen = build_dataloaders(args)

    for imgs, labels in testgen:
        imgs = sinv(imgs)

        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)

        fns = []
        for i in range(imgs.shape[0]):
            image = imgs[i].detach()
            ndarr = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            if ndarr.shape[2] == 1:
                ndarr = ndarr[:, :, 0]
            im = Image.fromarray(ndarr)

            md5hash = hashlib.md5(im.tobytes())
            fn = md5hash.hexdigest()

            if args.name:
                print(str(fn) + "," + classes[labels[i].item()])
            else:
                print(str(fn) + "," + str(labels[i].item()))



if __name__ == '__main__':
    main()