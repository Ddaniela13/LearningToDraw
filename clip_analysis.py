import argparse
import csv
import json
import pathlib

import clip
import torch
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-inp", help='path to full gamelogs', required=True, type=pathlib.Path)
    parser.add_argument("-out", help='path to save analysis', required=False, default=None, type=pathlib.Path)
    parser.add_argument("-hcm", help='path to hashed class mapping', required=False, type=pathlib.Path,
                        default=pathlib.Path("stl-test-image-classes-names-224.csv"))

    args = parser.parse_args()

    if args.out is None:
        args.out = args.inp / "clip-analysis.json"

    hashes_to_classes = {}
    with open(args.hcm) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for hsh, clz in reader:
            hashes_to_classes[hsh] = clz

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    stl_10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    templates = ["a photo of a {}.", "a drawing of a {}."]
    # Prepare the inputs

    text_inputs = []
    for template in templates:
        text_inputs.append(torch.cat([clip.tokenize(template.format(c)) for c in stl_10_classes]).to(device))
    text_inputs = torch.cat(text_inputs)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with open(args.inp / 'games.json') as f:
        data = json.load(f)

    output = []
    for i, game in enumerate(tqdm(data)):
        sketch_image = preprocess(Image.open(args.inp / game['sketch'])).unsqueeze(0).to(device)
        correct_image = preprocess(Image.open(args.inp / game['images'][game['correct_image']])).unsqueeze(0).to(device)
        guessed_image = preprocess(Image.open(args.inp / game['images'][game['reciever_guess']])).unsqueeze(0).to(
            device)

        # Calculate features
        with torch.no_grad():
            sketch_image_features = model.encode_image(sketch_image)
            correct_image_features = model.encode_image(correct_image)
            guessed_image_features = model.encode_image(guessed_image)

        # Pick the top 5 most similar labels for the image
        sketch_image_features /= sketch_image_features.norm(dim=-1, keepdim=True)
        correct_image_features /= correct_image_features.norm(dim=-1, keepdim=True)
        guessed_image_features /= guessed_image_features.norm(dim=-1, keepdim=True)

        def get_class_template(image_features):
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(1)
            index = indices[0]
            class_index = index % 10
            template_index = index // 10

            return stl_10_classes[class_index], templates[template_index]

        sketch_class, sketch_template = get_class_template(sketch_image_features)
        correct_class, correct_template = get_class_template(correct_image_features)
        guessed_class, guessed_template = get_class_template(guessed_image_features)
        true_class = hashes_to_classes[pathlib.Path(game['images'][game['correct_image']]).stem]

        rawdata = {
            'sketch_class': sketch_class,
            'sketch_template': 'drawing' if 'drawing' in sketch_template else 'photo',
            'correct_class': correct_class,
            'correct_template': 'drawing' if 'drawing' in correct_template else 'photo',
            'guessed_class': guessed_class,
            'guessed_template': 'drawing' if 'drawing' in guessed_template else 'photo',
            'true_class': true_class
        }
        output.append(rawdata)

    with open(args.out, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()
