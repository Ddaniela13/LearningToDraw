<div align="center">
  
# Learning to Draw: Emergent Communication through Sketching

This is the __official__ code for the paper ["Learning to Draw: Emergent Communication through Sketching"](https://arxiv.org/abs/2106.02067).

<p align="center">
  <a href="https://arxiv.org/abs/2106.02067">ArXiv</a> •
  <a href="https://paperswithcode.com/paper/learning-to-draw-emergent-communication">Papers With Code</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#game-setups">Game setups</a> •
  <a href="#model-setup">Model setup</a> •
  <a href="#datasets">Datasets</a> 
</p>

<img src="model.svg"/>
</a>

</div>

## About

We demonstrate that it is possible for a communication channel based on line drawing to emerge between agents playing a visual referential communication game. Furthermore we show that with a simple additional self-supervised loss that the drawings the agent produces are interpretable by humans.

## Getting started

You'll need to install the required dependencies listed in [requirements.txt](requirements.txt). This includes installing the differentiable rasteriser from the [DifferentiableSketching](https://github.com/jonhare/DifferentiableSketching) repository, and the source version of [https://github.com/pytorchbearer/torchbearer](https://github.com/pytorchbearer/torchbearer):

    pip install git+https://github.com/jonhare/DifferentiableSketching.git
    pip install git+https://github.com/pytorchbearer/torchbearer.git
    pip install -r requirements.txt

Once the dependencies are installed, you can run the `commgame.py` script to train and test models:

    python commgame.py train [args]
    python commgame.py test [args]

For example, to train a pair of agents on the _original_ game using the STL10 dataset (which will be downloaded if required), you would run:

    python commgame.py train --dataset STL10 --output stl10-original-model --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --batch-size 100

The options `--sigma2` and `--nlines` control the thickness and number of lines respectively. `--imagenet-weights` uses the standard pretrained imagenet vgg16 weights (use `--sin-weights` for stylized imagenet weights). Finally, `--freeze-vgg` freezes the backbone CNN, `--imagenet-norm` specifies to apply the imagenet normalisation to images (this should be used when using either imagenet or stylized imagenet weights), and `--invert` draws black strokes on a white canvas.

The training scripts compute a running communication rate in addition to loss and this is displayed as training progresses. After each epoch a validation pass is performed and images of the sketches and sender inputs and receiver targets are saved to the output directory along with a model snapshot. The output directory also contains a log file with the training and validation statistics per epoch.

Example commands to run the experiments in the paper are given in [commands.md](commands.md)

Further details on commandline arguments are given below.

## Game setups

All the setups involve a referential game where the reciever tries to select the "correct" image from a pool on the 
basis of a "sketch" provided by the sender.  The primary measure of success is the communication rate. The different command line arguments to control the different game variants are listed in the following subsections:


### Havrylov and Titov's Original Game Setup

Sender sees one image; Reciever sees many, where one is exactly the same as sender. 

Number of reciever images (target + distractors) is controlled by the batch-size. Number of sender images per iteration 
can also be controlled for completeness, but defaults to the same as batch size (e.g. each forward pass with a batch 
plays all possible game combinations using each of the images as a target).

    arguments:
    --batch-size
    [--sender-images-per-iter]


### Object-oriented Game Setup (same)

Sender sees one image; Reciever sees many, where one is exactly the same as sender and the others are all of different 
classes.

    arguments:
    --object-oriented same
    [--num-targets]
    [--sender-images-per-iter]


### Object-oriented Game Setup (different)

Sender sees one image; Reciever sees many, each of different classes; one of the images is the same class as the sender,
but is a completely different image).

    arguments:
    --object-oriented different 
    [--num-targets]
    [--sender-images-per-iter]
    [--random-transform-sender]


## Model setup

### Sender
The "sender" consists of a backbone VGG16 CNN which translates the input image into a latent vector and a "decoder" with an 
MLP that projects the latent representation from the backbone to a set of drawing commands that are differentiably 
rendered into an image which is sent to the "reciever". 

The backbone can optionally be initialised with pretrained weight and also optionally frozen (except for the final linear projection). The backbone, including linear projection can be shared between sender and reciever (default) or separate (`--separate_encoders`).

    arguments:
    [--freeze-vgg]
    [--imagenet-weights --imagenet-norm] 
    [--sin-weights --imagenet-norm] 
    [--separate_encoders]


### Receiver

The "receiver" consists of a backbone CNN which is used to convert visual inputs (both the images in the pool and the 
sketch) into a latent vector which is then transformed into a different latent representation by an MLP. These projected 
latent vectors are used for prediction and in the loss as described below. 

The actual backbone CNN model architecture will be the same as the sender's. The backbone can optionally share 
parameters with the "sender" agent. Alternatively it can be initialised with pre-trained weights, and also optionally frozen. 

    arguments:
    [--freeze-vgg]
    [--imagenet-weights --imagenet-norm]
    [--separate_encoders]


## Datasets

- MNIST
- CIFAR-10 / CIFAR-100
- TinyImageNet
- CelebA (`--image-size` to control size; default 64px)
- STL-10 
- Caltech101 (training data is balanced by supersampling with augmentation)

Datasets will be downloaded to the dataset root directory (default `./data`) as required.

    arguments: 
    --dataset {CIFAR10,CelebA,MNIST,STL10,TinyImageNet,Caltech101}  
    [--dataset-root]

## Citation

If you find this repository useful for your research, please cite our paper using the following.

      @@inproceedings{
      mihai2021learning,
      title={Learning to Draw: Emergent Communication through Sketching},
      author={Daniela Mihai and Jonathon Hare},
      booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
      year={2021},
      url={https://openreview.net/forum?id=YIyYkoJX2eA}
      }

