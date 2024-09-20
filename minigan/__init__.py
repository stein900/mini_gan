"""Mini-GAN: a compact DCGAN / conditional GAN on MNIST with classifier-based evaluation."""

from minigan.models import Discriminator, Generator, ModelConfig

__all__ = ["Discriminator", "Generator", "ModelConfig"]
__version__ = "0.2.0"
