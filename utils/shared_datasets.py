import collections

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10, CelebA, VisionDataset

# noinspection PyUnresolvedReferences
from dsketch.experiments.shared.args_datasets import *
from dsketch.experiments.shared.args_datasets import MNISTDataset as _MNISTDataset
from dsketch.experiments.shared.args_datasets import _Dataset, _split
from dsketch.experiments.shared.utils import list_class_names
from utils import *
from utils.caltech_datasets import Caltech101
from utils.tinyimagenet import TinyImageNet
import os
from torchvision import datasets
from skimage import io, transform

IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
IMAGENET_NORM_INV = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1. / 0.229, 1. / 0.224, 1. / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
])


class Balancing(Dataset):
    def __init__(self, dataset, transform=None):
        self.inner = dataset
        self.indices = []
        self.transform = transform

        # distribution of classes in the dataset
        original_mapping = {}
        label_to_count = {}
        for idx in range(len(dataset)):
            label = dataset[idx][1]
            label_to_count[label] = label_to_count.get(label, 0) + 1
            if label not in original_mapping:
                original_mapping[label] = collections.deque()
            original_mapping[label].append(idx)

        self.length = max(label_to_count.values()) * len(label_to_count.keys())
        for _ in range(max(label_to_count.values())):
            for clz in label_to_count.keys():
                idx = original_mapping[clz][0]
                self.indices.append(idx)
                original_mapping[clz].rotate(1)

    def __getitem__(self, index):
        img, lbl = self.inner[self.indices[index]]

        if self.transform:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return self.length


class Transforming(Dataset):
    def __init__(self, dataset, transform=None):
        self.inner = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, lbl = self.inner[index]

        if self.transform:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return len(self.inner)


random_seed = 1
torch.manual_seed(random_seed)


def compose(tf, args):
    if 'additional_transforms' in args and args.additional_transforms is not None:
        if tf is None:
            return args.additional_transforms
        return transforms.Compose([tf, args.additional_transforms])
    else:
        return tf


class MNISTDataset(_MNISTDataset):
    @staticmethod
    def num_classes():
        return 10


class CIFAR10Dataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--valset-size-per-class", help="number of examples to use in validation set per class",
                       type=int, default=10, required=False)
        p.add_argument("--dataset-seed", help="random seed for the train/validation split", type=int,
                       default=1234, required=False)
        p.add_argument("--imagenet-norm", help="normalise data with imagenet statistics", action='store_true',
                       required=False)

    @classmethod
    def get_transforms(cls, args, train=False):
        if args.imagenet_norm:
            tf = [transforms.ToTensor(), IMAGENET_NORM]
            return compose(transforms.Compose(tf), args)
        else:
            return compose(transforms.ToTensor(), args)

    @classmethod
    def get_size(cls, args):
        return 32
    
    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def create(cls, args):
        trainset = CIFAR10(args.dataset_root, train=True, transform=cls.get_transforms(args, True), download=True)
        testset = CIFAR10(args.dataset_root, train=False, transform=cls.get_transforms(args, False), download=True)

        train, valid = _split(args, trainset)

        return train, valid, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 10


class TinyImageNetDataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--imagenet-norm", help="normalise data with imagenet statistics", action='store_true',
                       required=False)

    @classmethod
    def get_transforms(cls, args, train=False):
        if args.imagenet_norm:
            return compose(transforms.Compose([transforms.ToTensor(), IMAGENET_NORM]), args)
        else:
            return compose(transforms.ToTensor(), args)

    @classmethod
    def get_size(cls, args):
        return 64

    @classmethod
    def get_channels(cls, args):
        return 3
    
    @classmethod
    def create(cls, args):
        trainset = TinyImageNet(args.dataset_root, split='train', transform=cls.get_transforms(args, True),
                                download=True)
        valset = TinyImageNet(args.dataset_root, split='val', transform=cls.get_transforms(args, False),
                              download=True)
        testset = TinyImageNet(args.dataset_root, split='test', transform=cls.get_transforms(args, False),
                               download=True)

        return trainset, valset, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 200


class STL10Dataset(_Dataset):
    @staticmethod
    def _add_args(p):
        CIFAR10Dataset._add_args(p)
        p.add_argument("--image-size", help="size of resampled images", type=int, default=96, required=False)
        p.add_argument("--unlabelled-train", help="load the unlabelled trainset", action='store_true', required=False)

    @classmethod
    def get_transforms(cls, args, train=False):
        base = [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor()
        ]

        if args.imagenet_norm:
            base.append(IMAGENET_NORM)

        return compose(transforms.Compose(base), args)

    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def get_size(cls, args):
        return args.image_size

    @classmethod
    def create(cls, args):
        tr = 'unlabelled' if args.unlabelled_train else 'train'
        trainset = STL10(args.dataset_root, split=tr, transform=cls.get_transforms(args, True), download=True)
        testset = STL10(args.dataset_root, split='test', transform=cls.get_transforms(args, False),
                        download=True)

        train, valid = _split(args, trainset)

        return train, valid, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 10


class CelebADataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--image-size", help="size of resampled images", type=int, default=64, required=False)
        p.add_argument("--imagenet-norm", help="normalise data with imagenet statistics", action='store_true',
                       required=False)

    @classmethod
    def get_transforms(cls, args, train=False):
        base = [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor()
        ]

        if args.imagenet_norm:
            base.append(IMAGENET_NORM)

        return compose(transforms.Compose(base), args)

    @classmethod
    def get_size(cls, args):
        return args.image_size

    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def create(cls, args):
        trainset = CelebA(args.dataset_root, split='train', target_type='identity',
                          transform=cls.get_transforms(args, True), download=True)
        valset = CelebA(args.dataset_root, split='valid', target_type='identity',
                        transform=cls.get_transforms(args, False), download=True)
        testset = CelebA(args.dataset_root, split='test', target_type='identity',
                         transform=cls.get_transforms(args, False), download=True)

        return trainset, valset, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 10177


class Caltech101Dataset(_Dataset):
    @staticmethod
    def _add_args(p):
        p.add_argument("--dataset-root", help="location of the dataset", type=pathlib.Path,
                       default=pathlib.Path("./data/"), required=False)
        p.add_argument("--image-size", help="size of resampled images", type=int, default=64, required=False)
        p.add_argument("--imagenet-norm", help="normalise data with imagenet statistics", action='store_true',
                       required=False)

    @classmethod
    def get_transforms(cls, args, train=False):

        if train:
            base = [
                transforms.Resize(args.image_size + 10),
                transforms.RandomCrop(args.image_size),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                                       p=0.8)
            ]
        else:
            base = [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor()
            ]

        if args.imagenet_norm:
            base.append(IMAGENET_NORM)

        return compose(transforms.Compose(base), args)

    @classmethod
    def get_size(cls, args):
        return args.image_size

    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def create(cls, args):
        caltechds = Caltech101(args.dataset_root, download=True)
        #         trainset =  Caltech101(args.dataset_root, split='train', transform=cls.get_transforms(args, True),
        #                                 download=True)
        #         valset = Caltech101(args.dataset_root, split='val', transform=cls.get_transforms(args, False),
        #                               download=True)
        #         testset = Caltech101(args.dataset_root, split='test', transform=cls.get_transforms(args, False),
        #                                download=True)

        print(len(caltechds))  # 8677 items
        y = caltechds.y
        trainset, x_test, y_train, y_test = train_test_split(caltechds, y, test_size=0.3, stratify=y, random_state=42)
        valset, testset, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test,
                                                          random_state=42)
        print(len(trainset))  # 6073
        print(len(valset))  # 1302
        print(len(testset))  # 1302
        #         trainset, valset, testset = torch.utils.data.random_split(caltechds, [6000, 1338, 1339])  # just some random numbers

        trainset = Balancing(trainset, transform=cls.get_transforms(args, True))
        valset = Transforming(valset, transform=cls.get_transforms(args, False))
        testset = Transforming(testset, transform=cls.get_transforms(args, False))

        return trainset, valset, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return 101


class FakeLabelledData(VisionDataset):
    def __init__(
            self,
            size: int = 10000,
            image_size=(3, 64, 64),
            num_classes: int = 10,
            transform=None
    ) -> None:
        super(FakeLabelledData, self).__init__(None)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        img = torch.rand(*self.image_size)
        img[0, :, :] = target
        img[1, :, :] = index
        torch.set_rng_state(rng_state)

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return self.size


class _TestingDataset(_Dataset):

    @staticmethod
    def _add_args(p):
        p.add_argument("--image-size", help="size of resampled images", type=int, default=64, required=False)
        p.add_argument("--num-classes", help="number of classes", type=int, default=10, required=False)

    @classmethod
    def get_transforms(cls, args, train=False):
        return compose(None, args)

    @classmethod
    def get_size(cls, args):
        return args.image_size

    @classmethod
    def get_channels(cls, args):
        return 3

    @classmethod
    def create(cls, args):
        sz = (3, args.image_size, args.image_size)
        nc = args.num_classes
        trainset = FakeLabelledData(100 * nc, sz, nc, transform=cls.get_transforms(args, True))
        valset = FakeLabelledData(10 * nc, sz, nc, transform=cls.get_transforms(args, False))
        testset = FakeLabelledData(10 * nc, sz, nc, transform=cls.get_transforms(args, False))
        return trainset, valset, testset

    @classmethod
    def inv_transform(cls, x):
        return x

    @staticmethod
    def num_classes():
        return None


def get_dataset(name):
    ds = getattr(sys.modules[__name__], name + 'Dataset')
    if not issubclass(ds, _Dataset):
        raise TypeError()
    return ds


def dataset_choices():
    return [i.replace('Dataset', '') for i in list_class_names(_Dataset, __name__)]


