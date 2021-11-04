import argparse
import base64
import json
import pathlib


def read_image(img):
    with open(img, "rb") as file:
        return "data:image/png;base64," + base64.b64encode(file.read()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-inp", help='path to full gamelogs', required=True, type=pathlib.Path)
    parser.add_argument("-out", help='path to html with rendering of game',
                        required=True, default=None, type=pathlib.Path)
    parser.add_argument("-id", type=int, required=False, help="game id to sample", default=0)

    args = parser.parse_args()

    with open(args.inp / 'games.json') as f:
        data = json.load(f)

    game = data[args.id]

    html = f"<html><head></head><body>"

    iimg = args.inp / game['images'][game['correct_image']]
    html += f"<div>Sender image:<br/> <img src='{read_image(iimg)}'></img></div>"
    html += "<br/>"

    isk = args.inp / game['sketch']
    html += f"<div>Sketch:<br/> <img src='{read_image(isk)}'></img></div>"
    html += "<br/>"

    html += "<div>Receiver images:<br/>"
    for tgt in game['images']:
        itgt = args.inp / tgt
        html += f"<img src='{read_image(itgt)}'></img>"
    html += "</div>"
    html += "</body></html>"

    with open(args.out, "w") as file:
        file.write(html)


if __name__ == '__main__':
    main()
