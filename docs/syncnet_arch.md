# Customize the architecture of SyncNet

The config file of SyncNet defines the architectures of audio and visual encoders. Let's first look at an example of an audio encoder:

```yaml
audio_encoder: # input (1, 80, 52)
  in_channels: 1
  block_out_channels: [32, 64, 128, 256, 512, 1024, 2048]
  downsample_factors: [[2, 1], 2, 2, 1, 2, 2, [2, 3]]
  attn_blocks: [0, 0, 0, 1, 1, 0, 0]
  dropout: 0.0
```

The above model architecture accepts a `1 x 80 x 52` image (mel spectrogram) and outputs a `2048 x 1 x 1` feature map. If the resolution of input image changes, you need to redefine the `downsample_factors` to make the output looks like `D x 1 x 1`, so that it can be used to compute cosine similarity.  Also reset the `block_out_channels`, in most cases, deeper networks require larger numbers of channels to store more features. We recommend reading the paper [EfficientNet](https://arxiv.org/abs/1905.11946), which discusses how to set the depth and width of CNN networks balancely. The `attn_blocks` defines whether a certain layer has a self-attention layer, where 1 indicates presence and 0 indicates absence. 

Now we look at an example of a visual encoder:

```yaml
visual_encoder: # input (48, 128, 256)
  in_channels: 48 # (16 x 3)
  block_out_channels: [64, 128, 256, 256, 512, 1024, 2048, 2048]
  downsample_factors: [[1, 2], 2, 2, 2, 2, 2, 2, 2]
  attn_blocks: [0, 0, 0, 0, 1, 1, 0, 0]
  dropout: 0.0
```

It is important to note that `in_channels`: it equals `num_frames * image_channels`. For pixel-space SyncNet, `image_channels` is 3, while for latent-space SyncNet, `image_channels` equals the `latent_channels` of the VAE you are using, typically 4 (SD 1.5, SDXL) or 16 (FLUX, SD3). In the example above, the visual encoder has an input frame length of 16 and is a pixel-space SyncNet, so `in_channels` is `16 x 3 = 48`.