# Audio Autoencoder
## Motivation
Motivated by a teacher on a online summer school, we tried to use an AutoEncoder for different purposes. Namely, noise reduction, timbre modification and audio separation

## Description
On this project we used datasets of piano and guitar sounds. We managed to get 100 samples of each. Since we are working with audio, we increased these sets by shuffling parts of sounds and re-pitching the instrument.

## Build Status
New ideas to try in the future.

## Files
*DATA*
- Guitar Loops, Piano Loops: Both of these are the original 100 samples we encountered. The augmented set has more than 1GB so it couldn't be uploaded.

*CODE*
- _AutoEncoderConv.py_: implementation of a Convolutional autoencoder in pytorch
- _SoundsDSConv.py_: used to load the sounds dataset in pytorch
- _piano_params.pth_, _guitar_params.pth_: parameters for the trained autoencoder

- _main.ipynb_: Main notebook where the tests are performed.

## Packages
- torch
- time
- torchaudio
- os
- pandas
- matplotlib

## Encountered difficulties:
  - Mean Squared Error loss gave us noisy outputs when we computed the inverse transformm of the spectrogram. Mean Absolute Error is a better choice of loss function for this particular task;
  - The autoencoder doesn't apply the constraint of non-negative values in the spectrogram, therefore clipping was applied to the networks output.
