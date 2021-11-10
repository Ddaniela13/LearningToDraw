# Commands to run the experiments from the paper
This file includes a list of example commands to train and evaluate our models to help reproduce the experiments from the paper:

- Original game setting, <img src="https://render.githubusercontent.com/render/math?math=l=l_{game}">

      python commgame.py train --dataset STL10 --output stl10-original-model-lgame --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3  --batch-size 100 --data-seed 0

      python commgame.py eval --dataset STL10 --output stl10-original-model-lgame-eval --loss HingeLoss --sigma2 5e-4 --nlines 20 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --invert --feature-layer relu5_3  --batch-size 100 --data-seed 0 --weights stl10-original-model-lgame/model_final.pt

- Original game setting, <img src="https://render.githubusercontent.com/render/math?math=l=l_{game}%2Bl_{perceptual}">

      python commgame.py train --dataset STL10 --output stl10-original-model-lpercep --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --sort-of-perceptual-loss --batch-size 100 --data-seed 0

      python commgame.py eval --dataset STL10 --output stl10-original-model-lpercep-eval --loss HingeLoss --sigma2 5e-4 --nlines 20 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --invert --feature-layer relu5_3 --sort-of-perceptual-loss --batch-size 100 --data-seed 0 --weights stl10-original-model-lpercep/model_final.pt

- Object-oriented *same*,  <img src="https://render.githubusercontent.com/render/math?math=l=l_{game}">

      python commgame.py train --dataset STL10 --output stl10-oosame-model-lgame --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --object-oriented same --batch-size 100 --data-seed 0
      
      python commgame.py eval --dataset STL10  --output stl10-oosame-model-lgame-eval --loss HingeLoss --sigma2 5e-4 --nlines 20 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --invert --feature-layer relu5_3 --object-oriented same --batch-size 100 --data-seed 0 --weights stl10-oosame-model-lgame/model_final.pt

- Object-oriented *different*, <img src="https://render.githubusercontent.com/render/math?math=l=l_{game}">

      python commgame.py train --dataset STL10 --output stl10-oodiff-model-lgame --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --object-oriented different --batch-size 100 --data-seed 0
    
      python commgame.py eval --dataset STL10  --output stl10-oodiff-model-lgame-eval --loss HingeLoss --sigma2 5e-4 --nlines 20 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --invert --feature-layer relu5_3 --object-oriented different --batch-size 100 --data-seed 0 --weights stl10-oodiff-model-lgame/model_final.pt

For the object-oriented games with  the objective <img src="https://render.githubusercontent.com/render/math?math=l=l_{game}%2Bl_{perceptual}">, just add the --sort-of-perceptual-loss argument to the train/eval commands above.


- Weighting the feature maps that contribute to the perceptual loss - example of using only the features out of the first convolution block:
      
      python commgame.py train --dataset STL10 --output stl10-original-sopweights10000 --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --sort-of-perceptual-loss --data-seed 0 --batch-size 100 --sop-weights 1 0 0 0 0 

- Varying model capacity, example on Caltech-101 128px images:

      python commgame.py train --dataset Caltech101 --output caltech101-oodiff-model-wide --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --imagenet-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --object-oriented different --sort-of-perceptual-loss --latent-size 1024 --rx-hidden 1024 --decoder-hidden 1024 --decoder-hidden2 1024 --weight-decay 0.0001 --batch-size 100 --data-seed 0

- Inducing shape bias into the model by using the weights of a VGG-16 pretrained on Stylized ImageNet (instead of the --imagenet, use --sin-weights):

      python commgame.py train --dataset CelebA --output celebA-model-original-sinweights --loss HingeLoss --sigma2 5e-4 --nlines 20 --learning-rate 0.0001 --dataset-root ./data --sin-weights --freeze-vgg --imagenet-norm --epochs 250 --invert --feature-layer relu5_3 --sort-of-perceptual-loss --data-seed 0 --batch-size 100 --image-size 112

Some general arguments you can control:

    [--sort-of-perceptual-loss] to use the additional perceptual loss
    [--sop-weights] to specify sort-of-perceptual-loss weights for the different feature maps, default is [1, 1, 1, 1, 1]
    [--sop-coef] to control the coeffiecient of the perceptual loss
    [--device]
    [--data-seed]
    [--snapshot-interval]
    [--num-reconstructions]
