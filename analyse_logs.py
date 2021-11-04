import argparse
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-inp", help='path to full gamelogs', required=True, type=pathlib.Path)
    args = parser.parse_args()

    with open(args.inp / 'games.json') as f:
        gamedata = json.load(f)

    with open(args.inp / 'clip-analysis.json') as f:
        clipdata = json.load(f)

    counters = {
        'commrate': [],
        'sketch_class_matches_true': [],
        'sketch_class_matches_correct': [],
        'sketch_class_matches_correct_and_true': [],
        'sketch_class_matches_guess': [],
        'sketch_class_matches_guess_and_true': [],
        'correct_class_matches_true': [],
        'guess_class_matches_true': [],
        'sketch_is_drawing': [],
        'correct_is_photo': [],
        'guess_is_photo': []
    }
    for game, clip in zip(gamedata, clipdata):
        counters['commrate'].append(1 if game['reciever_guess_correct'] else 0)
        counters['sketch_class_matches_true'].append(1 if clip['sketch_class'] == clip['true_class'] else 0)
        counters['sketch_class_matches_correct'].append(1 if clip['sketch_class'] == clip['correct_class'] else 0)
        counters['sketch_class_matches_correct_and_true'].append(
            1 if clip['sketch_class'] == clip['correct_class'] and clip['sketch_class'] == clip['true_class'] else 0)
        counters['sketch_class_matches_guess'].append(1 if clip['sketch_class'] == clip['guessed_class'] else 0)
        counters['sketch_class_matches_guess_and_true'].append(
            1 if clip['sketch_class'] == clip['guessed_class'] and clip['sketch_class'] == clip['true_class'] else 0)

        counters['correct_class_matches_true'].append(1 if clip['correct_class'] == clip['true_class'] else 0)
        counters['guess_class_matches_true'].append(1 if clip['guessed_class'] == clip['true_class'] else 0)

        counters['sketch_is_drawing'].append(1 if clip['sketch_template'] == 'drawing' else 0)
        counters['correct_is_photo'].append(1 if clip['correct_template'] == 'photo' else 0)
        counters['guess_is_photo'].append(1 if clip['guessed_template'] == 'photo' else 0)

    for k, v in counters.items():
        arr = np.array(v)
        print(k, sum(v), arr.mean())


if __name__ == '__main__':
    main()
