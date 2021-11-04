import argparse
import json
import pathlib

import torchvision
import torchvision.transforms.functional as F
from PIL import Image


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("input", help='path to full gamelogs', type=pathlib.Path, nargs='+')
    parser.add_argument("-out", help='path to output image', required=True, default=None, type=pathlib.Path)
    parser.add_argument("-num", type=int, required=False, help="number of sketches", default=100)
    parser.add_argument("-tout", help='path to output target image', required=True, default=None, type=pathlib.Path)

    args = parser.parse_args()

    res = dict()
    tgt = dict()
    counts = dict()
    for inp in args.input:
        with open(inp / 'games.json') as f:
            data = json.load(f)

            for i in range(len(data)):
                game = data[i]
                key = game['images'][game['correct_image']]

                if len(res) >= args.num and key not in res:
                    continue

                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
                print(counts)

                isk = inp / game['sketch']
                with Image.open(isk) as img:
                    timg = F.to_tensor(img)

                    if key not in res:
                        res[key] = timg
                    else:
                        res[key] += timg

                tgtimg = inp / key
                with Image.open(tgtimg) as img:
                    timg = F.to_tensor(img)

                    if key not in tgt:
                        tgt[key] = timg

    # keys = res.keys()[:args.num]
    # res = {key: res[key] for key in keys}
    # tgt = {key: tgt[key] for key in keys}

    for k, v in res.items():
        res[k] = v / len(args.input)

    torchvision.utils.save_image(list(res.values()), args.out, nrow=10)
    torchvision.utils.save_image(list(tgt.values()), args.tout, nrow=10)


if __name__ == '__main__':
    main()
