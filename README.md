# audio_autoencoder
We train an autoencoder to recognize certain instruments.

Encountered difficulties:
  - Mean Squared Error loss gave us noisy outputs when we computed the inverse transformm of the spectrogram. Mean Absolute Error is a better choice of loss function for this particular task;
  - The autoencoder doesn't apply the constraint of non-negative values in the spectrogram, therefore clipping was applied to the networks output.
