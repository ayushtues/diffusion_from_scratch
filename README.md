# Diffusion From Scratch
Teaching myself diffusion models by coding one from scratch

End goal is to be able to code something following the basic concepts of diffusion, train it, and hopefully sample something remotely resembling MNIST, given the compute I can get lol


## Features
- [x] Dataloader for MNIST
- [x] Sinusoidal time embedding
- [x] Linear Noise schedule
- [x] Simple UNet based architecture
- [x] Forward and reverse diffusion process with unlearnt variance
- [x] Sampling images
- [ ] Cosine noise schedule
- [ ] EMA for weights
- [ ] Learned variance
- [ ] Hybrid loss ( VB + Simple )
- [ ] Loss based importance sampling of time
- [ ] Actual Unet artchitecture from the paper, with attention

## Samples

Currently the model is making the noise increase and explode, need to figure out how to stabilize it 
