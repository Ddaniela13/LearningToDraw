import argparse
import pathlib
import json
import random
import shutil
import csv


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-inp", help='path to full gamelogs', required=True, type=pathlib.Path)
    parser.add_argument("-out", help='path to sampled gamelogs. Defaults to the input path with "-sampled" appended', 
        required=False, default=None, type=pathlib.Path)
    parser.add_argument("-n", type=int, required=False, help="number of samples", default=30)
    
    args = parser.parse_args()

    if args.out is None: 
        args.out = args.inp.parent / (args.inp.name + "-sampled")

    print(" Input folder: ", args.inp)
    print("Output folder: ", args.out)
    print(" Num. samples: ", args.n)

    args.out.mkdir(parents=True)

    sketchdir = args.out / "sketch-images"
    sketchdir.mkdir(parents=True)
    targetdir = args.out / "target-images"
    targetdir.mkdir(parents=True)

    with open(args.inp / 'games.json') as f:
        data = json.load(f)

    sample = random.sample(data, args.n)

    with open(args.out / 'games.json', 'w') as f:
        json.dump(sample, f)

    with open(args.out / 'games.csv', 'w', encoding='utf8', newline='') as f:
        fc = csv.DictWriter(f, fieldnames=sample[0].keys())
        fc.writeheader()
        fc.writerows(sample)

    for game in sample:
        isk = args.inp / game['sketch']
        osk = args.out / game['sketch']
        shutil.copy(str(isk), str(osk))

        for tgt in game['images']:
            itgt = args.inp / tgt
            otgt = args.out / tgt
            shutil.copy(str(itgt), str(otgt))


if __name__ == '__main__':
    main()