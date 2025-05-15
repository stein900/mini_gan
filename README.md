# Mini-GAN

DCGAN and a class-conditional GAN on MNIST. Samples are scored with a small
MNIST classifier: Frechet distance and an inception score in that feature
space, plus a nearest-neighbour check against the training set.

The reference DCGAN (seed 0, 50 epochs) is in `weights/dcgan_generator.pt`.
The conditional model is in `weights/cgan_generator.pt`.

## License

MIT. See LICENSE.
