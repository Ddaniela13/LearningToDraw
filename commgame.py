import argparse
import pathlib
from itertools import chain

import numpy as np
import torch
import torchvision.transforms as transforms

import torchbearer as tb
import torchbearer.metrics as metrics
from dsketch.experiments.characters.models import Encoder, Decoder
from dsketch.experiments.shared.utils import img_to_file, get_subparser, save_args, save_model_info, \
    FakeArgumentParser, parse_learning_rate_arg
from model import SketchingGame, get_model, model_choices, SKETCHES, SENDER_IMAGES, RECEIVER_IMAGES, \
    RECEIVER_IMAGES_MATCHED
from model.agents import _RxBase
from torchbearer.callbacks import Interval, CSVLogger, imaging, add_to_loss, on_end_epoch
from utils.game_data import build_game_loader, build_dataloaders, pair_images, pair_images_tranform_sender
from utils.game_logging import GameLogger
from utils.game_loss import loss_choices, get_loss as get_loss, build_loss, reconstruct_loss, \
    LearnableWeightedPerceptualLoss, make_perceptual_loss, make_CLIP_loss, make_dog_perceptual_loss
from utils.shared_datasets import dataset_choices, get_dataset, IMAGENET_NORM_INV


def build_commrate_metric(args):
    @metrics.running_mean
    @metrics.std
    @metrics.mean
    @metrics.lambda_metric('commrate', on_epoch=False)
    def commrate(y_pred, y_true):
        if "games_per_batch" not in args:
            tmp = y_pred.argmax(dim=1)
            return (tmp == y_true).float()
        else:
            correct = []
            gamesize = y_pred.shape[0] // args.games_per_batch

            for i in range(0, y_pred.shape[0], gamesize):
                _y_pred = y_pred[i: i + gamesize]
                _y_true = y_true[i: i + gamesize] - i
                tmp = _y_pred.argmax(dim=1)
                correct.append((tmp == _y_true).float())
            return torch.cat(correct, dim=0)

    return commrate


def train(args):
    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output, name='train_cmd.txt')

    traingen, valgen, testgen = build_dataloaders(args)

    trial, model = build_trial(args)
    save_model_info(model, args.output)

    trial.to(args.device)
    trial.with_generators(train_generator=traingen, val_generator=valgen, test_generator=testgen)

    trial.run(epochs=args.epochs, verbose=2)

    state = trial.state_dict()
    string_state = {str(key): state[key] for key in state.keys()}
    torch.save(string_state, str(args.output) + '/model_final.pt')
    
    
def evaluate(args):
    args.output.mkdir(exist_ok=True, parents=True)
    save_args(args.output, name='eval_cmd.txt')

    traingen, valgen, testgen = build_dataloaders(args)

    trial, model = build_test_trial(args)
    model.to(args.device)
    save_model_info(model, args.output)

    trial.to(args.device)
    trial.with_generators(test_generator=testgen)
    trial.predict(verbose=2)
    

def build_trial(args):
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    
    model = build_model(args)
    loss = build_loss(args)

    if args.sort_of_perceptual_loss and args.learn_sop_weights:
        ls = LearnableWeightedPerceptualLoss().to(args.device)
        params = filter(lambda p: p.requires_grad, chain(model.parameters(), ls.parameters()))
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())

    init_lr, sched = parse_learning_rate_arg(args.learning_rate)
    optim = torch.optim.Adam(params, lr=init_lr, weight_decay=args.weight_decay)

    inv = get_dataset(args.dataset).inv_transform

    if 'imagenet_norm' in args and args.imagenet_norm:
        sinv = transforms.Compose([inv, IMAGENET_NORM_INV])
    else:
        sinv = inv

    callbacks = [
        Interval(filepath=str(args.output) + '/model_{epoch:02d}.pt', period=args.snapshot_interval),
        CSVLogger(str(args.output) + '/log.csv'),
        imaging.FromState(SKETCHES, transform=inv).on_val().cache(args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_reconstruction_samples_{epoch:02d}.png')),
        imaging.FromState(SENDER_IMAGES, transform=sinv).on_val().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_samples.png')),
        imaging.FromState(RECEIVER_IMAGES, transform=sinv).on_val().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_samples_targets_gameorder.png')),
        imaging.FromState(RECEIVER_IMAGES_MATCHED, transform=sinv).on_val().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/val_samples_targets_matchedorder.png')),
        *model.get_callbacks(args),
        *sched
    ]

    if args.reconstruct_loss:
        callbacks.append(reconstruct_loss)

    if args.sort_of_perceptual_loss:
        if args.learn_sop_weights:
            @add_to_loss
            def lpl(state):
                return ls(state)

            @on_end_epoch
            def print_weights(state):
                print(ls.weights)

            callbacks.append(lpl)
            callbacks.append(print_weights)
        else:
            callbacks.append(make_perceptual_loss(args.sop_weights, args.sop_coef))

    if args.like_a_dog_loss:
        callbacks.append(make_dog_perceptual_loss(args.sop_weights, args.sop_coef, args.like_a_dog_image))

    if args.CLIP_loss:
        callbacks.append(make_CLIP_loss(args))

    trial = tb.Trial(model, optimizer=optim, criterion=loss, metrics=['loss', build_commrate_metric(args), 'lr'],
                     callbacks=callbacks)
    trial.with_loader(build_game_loader(args))

    return trial, model    


def build_test_trial(args):
    model = build_model(args).to(args.device)
    loss = build_loss(args)
    inv = get_dataset(args.dataset).inv_transform

    if 'imagenet_norm' in args and args.imagenet_norm:
        sinv = transforms.Compose([inv, IMAGENET_NORM_INV])
    else:
        sinv = inv

    callbacks = [
        CSVLogger(str(args.output) + '/log.csv'),
        imaging.FromState(SKETCHES, transform=inv).on_test().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_sketch_samples.png')),
        imaging.FromState(SENDER_IMAGES, transform=sinv).on_test().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_samples.png')),
        imaging.FromState(RECEIVER_IMAGES, transform=sinv).on_test().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_samples_targets_gameorder.png')),
        imaging.FromState(RECEIVER_IMAGES_MATCHED, transform=sinv).on_test().cache(
            args.num_reconstructions).make_grid().with_handler(
            img_to_file(str(args.output) + '/test_samples_targets_matchedorder.png')),
        GameLogger(args, transform=sinv, sketch_transform=inv)
    ]

    if args.reconstruct_loss:
        callbacks.append(reconstruct_loss)

    mets = ['loss', build_commrate_metric(args)]
    trial = tb.Trial(model, criterion=loss, metrics=mets, callbacks=callbacks)
    trial.with_loader(build_game_loader(args))

    return trial, model


def build_model(args):
    senc = get_model(args.encoder).create(args)
    if args.separate_encoders:
        renc = get_model(args.encoder).create(args)
    else:
        renc = senc

    dec = get_model(args.decoder).create(args)
    rec = get_model(args.receiver).create(args)

    imagenet_norm = args.imagenet_norm if 'imagenet_norm' in args else False
    sketching_agents = SketchingGame(senc, renc, dec, rec, args.num_targets, imagenet_norm, args.invert)

    if 'weights' in args:
        if args.swapAgents and args.weights_pair2 is not None:
                   
            print("Loading weights for the second pair of agents to swap")
            
            state1 =  torch.load(args.weights, map_location=args.device)
            sketching_agents.load_state_dict(state1[tb.MODEL])
            pair1_sender_enc=sketching_agents.sender_encoder
            pair1_dec = sketching_agents.decoder
            
            
            senc2 = get_model(args.encoder).create(args)
            if args.separate_encoders:
                renc2 = get_model(args.encoder).create(args)
            else:
                renc2 = senc2

            dec2 = get_model(args.decoder).create(args)
            rec2 = get_model(args.receiver).create(args)

            sketching_agents2 = SketchingGame(senc2, renc2, dec2, rec2, args.num_targets, imagenet_norm, args.invert)
            
            state2 =  torch.load(args.weights_pair2, map_location=args.device)
            sketching_agents2.load_state_dict(state2[tb.MODEL])
            pair2_rec_enc=sketching_agents2.receiver_encoder
            pair2_rec=sketching_agents2.receiver

            sketching_agents = SketchingGame(pair1_sender_enc, pair2_rec_enc, pair1_dec, pair2_rec, args.num_targets, imagenet_norm, args.invert)
        else:
            print("Weight loading for one pair of agents only")
            state = torch.load(args.weights, map_location=args.device)
            sketching_agents.load_state_dict(state[tb.MODEL])

    # TODO: loading weights for encoders

    if args.freeze_encoders:
        for p in renc.parameters():
            p.requires_grad = False
        for p in senc.parameters():
            p.requires_grad = False

    return sketching_agents


def add_shared_args(parser):
    parser.add_argument("--encoder", help="encoder class", required=False, default='VGG16BackboneExtended', choices=model_choices(Encoder))
    parser.add_argument("--separate-encoders", help="use non-shared encoder weights between sender/receiver",
                        action='store_true', required=False)
    parser.add_argument("--freeze-encoders", help="freeze encoder weights", action='store_true', required=False)
    parser.add_argument("--decoder", help="decoder class", required=False, default='SinglePassSimpleLineDecoder', choices=model_choices(Decoder))
    parser.add_argument("--receiver", help="receiver class", required=False, default='SimpleReceiverMLP', choices=model_choices(_RxBase))
    parser.add_argument("--dataset", help="dataset", required=True, choices=dataset_choices())
    parser.add_argument("--num-reconstructions", type=int, required=False, help="number of reconstructions to save",
                        default=100)
    parser.add_argument("--random-transform-sender", help='apply random transformation to sender images',
                        required=False, action='store_true')
    parser.add_argument("--sender-images-per-iter", help="number of sender images in each batch of processing",
                        type=int, default=None)
    parser.add_argument("--num-targets", help="number of target images in each game (only for OO games); defaults "
                                              "to the number of classes", type=int, default=None)
    parser.add_argument("--object-oriented", help='enable oo game variation', required=False,
                        choices=['same', 'different', 'mixed'], default=None)
    parser.add_argument("--output", help='folder to store outputs', required=True, type=pathlib.Path)
    parser.add_argument("--device", help='device to use', required=False, type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--data-seed", help='random seed for dataset shuffling, etc', required=False, type=int,
                        default=1234)
    parser.add_argument("--invert", action='store_true', required=False, help="should the decoded image be inverted "
                                                                              "before being passed to reciever")

    # TODO: this is a bit of a hack and perhaps needs more thought as to how its computed, but it does seem to work on
    # oo games
    parser.add_argument("--sort-of-perceptual-loss", action='store_true', required=False,
                        help="should a pseudo perceptual loss between the senders input and sketch be incorporated.")
    parser.add_argument("--like-a-dog-loss", action='store_true', required=False,
                        help="should a pseudo perceptual loss between an image of a dog and sketch be incorporated.")
    parser.add_argument("--like-a-dog-image", required=False, type=pathlib.Path,
                        help="path to an image of a dog or similar.")
    parser.add_argument("--learn-sop-weights", action='store_true', required=False,
                        help="should the sort-of-perceptual loss weights be learned?")
    parser.add_argument("--sop-weights", type=float, nargs='+', required=False, default=[1, 1, 1, 1, 1],
                        help="sort-of-perceptual loss weights")
    parser.add_argument("--sop-coef", type=float, required=False, help="weight the perceptual loss - lambda", default=1)
    parser.add_argument("--reconstruct-loss", help="Add a reconstruction loss between input image and sketch",
                        action='store_true',
                        required=False)
    parser.add_argument("--loss", choices=loss_choices(), required=True, help="loss function")
    parser.add_argument("--CLIP-loss", action='store_true', required=False,
                        help="incorporate the pseudo perceptual loss (from CLIPDraw) between the senders input and sketch according.")


def add_subparsers(parser, add_help=True):
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser("train", add_help=add_help)
    add_shared_args(train_parser)
    train_parser.add_argument("--epochs", type=int, required=False, help="number of epochs", default=10)
    train_parser.add_argument("--learning-rate", type=str, required=False, help="learning rate spec", default=0.001)
    train_parser.add_argument("--weight-decay", type=float, required=False, help="weight decay", default=0)
    train_parser.add_argument("--snapshot-interval", type=int, required=False,
                              help="interval between saving model snapshots", default=10)
    train_parser.set_defaults(perform=train)

    eval_parser = subparsers.add_parser("eval", add_help=add_help)
    add_shared_args(eval_parser)
    eval_parser.add_argument("--weights", help="model weights", type=pathlib.Path, required=True)
    eval_parser.add_argument("--swapAgents",  action='store_true', required=False,
                        help="Pair the sender of the firs pair at location --weights with the receiver of the second pair at location --weights-pair2")
    eval_parser.add_argument("--weights-pair2", help="second model weights", type=pathlib.Path, required=False)
    
    eval_parser.set_defaults(perform=evaluate)


def add_sub_args(args, parser):
    subparser = get_subparser(args.mode, parser)

    if 'encoder' in args and args.encoder is not None:
        get_model(args.encoder).add_args(subparser)
    if 'decoder' in args and args.decoder is not None:
        get_model(args.decoder).add_args(subparser)
    if 'receiver' in args and args.receiver is not None:
        get_model(args.receiver).add_args(subparser)

    if 'dataset' in args and args.dataset is not None:
        get_dataset(args.dataset).add_args(subparser)

    if args.mode == 'train' and 'loss' in args and args.loss is not None:
        get_loss(args.loss).add_args(subparser)


def parse_args(argslist=None):
    fake_parser = FakeArgumentParser(add_help=False, allow_abbrev=False)
    add_subparsers(fake_parser, add_help=False)
    fake_args, _ = fake_parser.parse_known_args(argslist)

    if fake_args.mode is None:
        print("Mode must be specifed {train,eval}.")
        return

    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_subparsers(parser)
    add_sub_args(fake_args, parser)

    args = parser.parse_args(argslist)
    args.channels = get_dataset(args.dataset).get_channels(args)

    return args


def main():
    args = parse_args()

    # this is where we configure how images are paired-up
    if args.random_transform_sender:
        # sender undergoes transform
        print(f"Sender images will be a randomly transformed; targets will not.")
        args.additional_transforms = pair_images_tranform_sender
    else:
        # straight pair - images unaltered
        print(f"Sender (and target) images are not transformed.")
        args.additional_transforms = pair_images

    args.perform(args)


if __name__ == '__main__':
    main()
