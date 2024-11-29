# Mini-GAN

DCGAN on MNIST, 28x28, latent size 100. Weights use the DCGAN normal
initialisation. Training is Adam with the DCGAN learning rate and betas.

```bash
python -m pip install -r requirements.txt
python train.py
```

Samples for a fixed latent batch are written each epoch.

## License

MIT. See LICENSE.
