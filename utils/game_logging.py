import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

import torchbearer as tb
from model import SKETCHES, RECEIVER_IMAGES
from torchbearer import Callback


class GameLogger(Callback):
    def __init__(self, args, transform=None, sketch_transform=None):
        self.transform = transform
        self.sketch_transform = sketch_transform
        self.args = args

        self.basedir = str(args.output) + "/gamelogs/"
        self.imagedir = "target-images"
        self.sketchdir = "sketch-images"
        Path(self.basedir + "/" + self.imagedir).mkdir(exist_ok=True, parents=True)
        Path(self.basedir + "/" + self.sketchdir).mkdir(exist_ok=True, parents=True)

        self.allgames = []

    def _save_images(self, imgs, sdir):
        if sdir == self.sketchdir:
            imgs = self.sketch_transform(imgs)
        else:
            imgs = self.transform(imgs)

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
            file = self.basedir + "/" + sdir + "/" + fn + ".png"
            fns.append(sdir + "/" + fn + ".png")
            if not Path(file).exists():
                im.save(file)

        return fns

    def _save_game(self, games):
        self.allgames.extend(games)

        with open(self.basedir + '/games.json', 'w') as fp:
            json.dump(self.allgames, fp)

        with open(self.basedir + '/games.csv', 'w', encoding='utf8', newline='') as output_file:
            fc = csv.DictWriter(output_file, fieldnames=self.allgames[0].keys())
            fc.writeheader()
            fc.writerows(self.allgames)

    def on_forward_test(self, state):
        self.on_forward_validation(state)

    def on_forward_validation(self, state):
        sketches = state[SKETCHES]
        receiver_images = state[RECEIVER_IMAGES]

        skids = self._save_images(sketches, self.sketchdir)
        imids = self._save_images(receiver_images, self.imagedir)

        y_pred = state[tb.Y_PRED]
        y_true = state[tb.Y_TRUE]

        games = []

        if "games_per_batch" not in self.args:
            correct = y_pred.argmax(dim=1) == y_true

            for i in range(len(correct)):
                games.append({
                    'reciever_guess': y_pred.argmax(dim=1)[i].cpu().item(),
                    'reciever_guess_correct': correct[i].cpu().item(),
                    'sketch': skids[i],
                    'correct_image': y_true[i].item(),
                    'images': imids
                })
        else:
            gamesize = y_pred.shape[0] // self.args.games_per_batch

            for j in range(0, y_pred.shape[0], gamesize):
                _imids = imids[j: j + gamesize]
                _skids = skids[j: j + gamesize]
                _y_pred = y_pred[j: j + gamesize]
                _y_true = y_true[j: j + gamesize] - j
                correct = _y_pred.argmax(dim=1) == _y_true

                for i in range(len(correct)):
                    games.append({
                        'reciever_guess': _y_pred.argmax(dim=1)[i].cpu().item(),
                        'reciever_guess_correct': correct[i].cpu().item(),
                        'sketch': _skids[i],
                        'correct_image': _y_true[i].item(),
                        'images': _imids
                    })

        self._save_game(games)
