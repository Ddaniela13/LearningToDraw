import collections

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms

import torchbearer
from dsketch.experiments.shared.utils import ORIGINAL_Y_TRUE
from model import SENDER_IMAGES, RECEIVER_IMAGES, RECEIVER_IMAGES_MATCHED
from utils.shared_datasets import get_dataset


def build_game_loader(args):
    # TorchBearer Loader that transforms the inputs into a tuple of batches with different orders, records the target
    # order in the Y_TRUE and stashes the original class labels for use if required later
    def game_loader(state):
        (sender_images, target_images), label = torchbearer.deep_to(next(state[torchbearer.ITERATOR]),
                                                                    state[torchbearer.DEVICE],
                                                                    state[torchbearer.DATA_TYPE])

        if args.sender_images_per_iter is None:
            nsender = sender_images.shape[0]
        else:
            nsender = args.sender_images_per_iter

        # compute the  permutation of the targets, and the corresponding conjugate permutation which provides the labels
        # for the inputs
        if "games_per_batch" in args:
            p = []
            imgs_per_game = sender_images.shape[0] // args.games_per_batch
            for i in range(0, sender_images.shape[0] // imgs_per_game):
                pp = torch.randperm(imgs_per_game, device=state[torchbearer.DEVICE])
                pp = pp + i * imgs_per_game
                p.append(pp)

            p = torch.cat(p, dim=0)
        else:
            p = torch.randperm(sender_images.shape[0], device=state[torchbearer.DEVICE])

        pinv = torch.argsort(p)[:nsender]

        sender_images = sender_images[:nsender]  # select desired number of inputs
        target_images_game_order = target_images[p]  # all targets used, but permuted

        state[torchbearer.X] = (sender_images, target_images_game_order)
        state[SENDER_IMAGES] = sender_images
        state[RECEIVER_IMAGES] = target_images_game_order  # these are shuffled for game play
        state[RECEIVER_IMAGES_MATCHED] = target_images  # these are ordered 1-1 with sender
        state[torchbearer.Y_TRUE] = pinv
        state[ORIGINAL_Y_TRUE] = label

    return game_loader


def compute_mapping(dataset):
    d = dict()

    for idx, (_, clz) in enumerate(dataset):
        clz = int(clz)
        if clz not in d:
            d[clz] = []
        d[clz].append(idx)

    return d


class ShuffledDataset(Dataset):
    def __init__(self, ds, generator=None):
        self.ds = ds

        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        self.indices = torch.randperm(len(ds), generator=generator).tolist()

    def __getitem__(self, index):
        return self.ds[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class GameSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_targets=None, batch_size=None, shuffle=False, generator=None, mode='same'):
        self.mapping = compute_mapping(dataset)
        self.generator = generator
        self.lengths = dict()
        self.length = 0
        self.nclz = len(self.mapping)
        self.shuffle = shuffle
        self.mode = mode

        for clz, idx in self.mapping.items():
            self.lengths[clz] = len(idx)
            self.length += self.lengths[clz]

        if num_targets is None:
            num_targets = self.nclz

        self.num_targets_per_game = min(num_targets, self.nclz)
        if batch_size is None:
            self.games_per_batch = 1
        else:
            self.games_per_batch = max(1, batch_size // self.num_targets_per_game)

    def __len__(self):
        return max(1, self.length // (self.games_per_batch * self.num_targets_per_game))

    def __iter__(self):
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        indices = dict()
        for clz, lng in self.lengths.items():
            if self.shuffle:
                indices[clz] = collections.deque(torch.randperm(lng, generator=generator).tolist())
            else:
                indices[clz] = collections.deque(list(range(lng)))

        batches = []
        if self.mode == 'same':
            for i in range(max(1, self.length // (self.games_per_batch * self.num_targets_per_game))):
                batch = []

                if self.shuffle:
                    clzs = list(self.mapping.keys())
                    clzorder = [clzs[k] for k in torch.randperm(self.nclz, generator=generator).tolist()]
                    clzorder = collections.deque(clzorder)
                else:
                    clzorder = collections.deque(list(self.mapping.keys()))

                for _ in range(self.games_per_batch):
                    for j in range(self.num_targets_per_game):
                        clz = clzorder[j]
                        batch.append(self.mapping[clz][indices[clz][0]])
                        indices[clz].rotate(1)
                    clzorder.rotate(self.num_targets_per_game)
                batches.append(batch)
        elif self.mode == 'different':
            # note the collate_oo_diffmixed will be used to effectively re-pair the items

            for i in range(max(1, self.length // (self.games_per_batch * self.num_targets_per_game))):
                batch = []

                if self.shuffle:
                    clzs = list(self.mapping.keys())
                    clzorder = [clzs[k] for k in torch.randperm(self.nclz, generator=generator).tolist()]
                    clzorder = collections.deque(clzorder)
                else:
                    clzorder = collections.deque(list(self.mapping.keys()))

                for _ in range(self.games_per_batch):
                    for j in range(self.num_targets_per_game):
                        clz = clzorder[j]
                        batch.append(self.mapping[clz][indices[clz][0]])
                        batch.append(self.mapping[clz][indices[clz][-1]])
                        indices[clz].rotate(1)
                    clzorder.rotate(self.num_targets_per_game)
                batches.append(batch)
        elif self.mode == 'mixed':
            raise NotImplementedError()

        yield from batches


def collate_oo_diffmixed(items):
    bs = len(items)

    newitems = []
    for i in range(0, bs, 2):
        # pair up the first sender image with the receiver one from +1 position which should have the same class
        newitems.append([(items[i][0][0], items[i + 1][0][1]), items[i][1]])
        assert items[i][1] == items[i + 1][1]

    col = default_collate(newitems)
    return col


def build_dataloaders(args, verbose=True):
    ds = get_dataset(args.dataset)
    args.size = ds.get_size(args)

    train, valid, test = ds.create(args)

    generator = torch.Generator()
    generator.manual_seed(args.data_seed)

    if args.object_oriented is not None:
        bs = args.batch_size
        num_targets = args.num_targets

        if args.sender_images_per_iter is not None:
            print("warning: using less sender images than the batch size stops reduces the size of the batch")
            bs = args.num_targets

        sampler = GameSampler(train, num_targets=num_targets, batch_size=bs, generator=generator,
                              shuffle=True, mode=args.object_oriented)

        args.games_per_batch = sampler.games_per_batch
        args.num_targets = sampler.num_targets_per_game
        args.batch_size = sampler.num_targets_per_game * sampler.games_per_batch
        if verbose:
            print(f"Using the object-oriented game setting")
            nsender = args.batch_size if args.sender_images_per_iter is None else args.sender_images_per_iter
            print(f"Using {nsender} sender images per iteration (=number of games/batch)")
            print(f"Each game has {sampler.num_targets_per_game - 1} distractor images")

        if args.object_oriented == 'same':
            collate_fn = None
        else:
            collate_fn = collate_oo_diffmixed

        trainloader = DataLoader(train, num_workers=args.num_workers, batch_sampler=sampler, collate_fn=collate_fn)

        valloader = DataLoader(valid, num_workers=args.num_workers,
                               batch_sampler=GameSampler(valid, generator=generator, shuffle=False,
                                                         num_targets=num_targets, batch_size=bs,
                                                         mode=args.object_oriented),
                               collate_fn=collate_fn)

        testloader = DataLoader(test, num_workers=args.num_workers,
                                batch_sampler=GameSampler(test, generator=generator, shuffle=False,
                                                          num_targets=num_targets, batch_size=bs,
                                                          mode=args.object_oriented),
                                collate_fn=collate_fn)
    else:
        if args.num_targets is not None:
            print("warning: --num-targets value is ignored in standard gameplay and determined by the batch size")

        trainloader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                 generator=generator)

        valid = ShuffledDataset(valid, generator=generator)
        test = ShuffledDataset(test, generator=generator)
        valloader = DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                               generator=generator)
        testloader = DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                generator=generator)

        if verbose:
            print(f"Using the standard game setting")
            nsender = args.batch_size if args.sender_images_per_iter is None else args.sender_images_per_iter
            print(f"Using {nsender} sender images per iteration (=number of games/batch)")
            print(f"Each game has {args.batch_size - 1} distractor images")

    return trainloader, valloader, testloader


def pair_images(x):
    return x, x


RANDOM_TF = transforms.Compose([
    transforms.RandomAffine(10.0, translate=(0.1, 0.1), scale=(0.95, 1.01), shear=1, fillcolor=None),
    transforms.Lambda(lambda x: x * (1 + (torch.rand_like(x) - 0.5) / 10))
])


def pair_images_tranform_sender(x, tf=RANDOM_TF):
    return tf(x), x

