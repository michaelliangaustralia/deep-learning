# Deep Learning

Monorepo for all deep learning related code intended purely for educational and experimental purposes.

## Projects

All project specific code exists in the `projects` folder.
- `audio_classification_freesound` - Freesound audio classification Kaggle challenge 2019.
- `asr_finetune_w2v2` - English ASR by finetuning Meta's Wav2Vec2 on various datasets.
- `tts_tacotron2` - Text to speech TacoTron 2.
- `rl_mad_mario` - Reinforcement learning to train an agent in Mario game.
- `np_rnn` - Build a RNN using numpy.
- `np_cnn` - Build a CNN using numpy.
- `pytorch_transformer` - Build a transformer using PyTorch.
- `binary_classification_spaceship_titanic` - Binary classification Kaggle competition.
- `predict_house_price` - Housing price prediction Kaggle competition.
- `gan_mnist` - Generative Adversial Networks using the MNIST dataset.
- `gan_monet` - Generative Adversial Networks Kaggle competition.
- `gan_pickle` - CycleGAN to turn merge any picture with the cutest dog.
- `stable_diffusion_2` - Stable diffusion 2 project.
- `voice_activity_detector` - Neural VAD classifying speech, noise and music.

## Common utils

`common` contains common utils that are used across projects.

```python
import sys
sys.path.append("../..")
import common.utils as common_utils
```

## Comet Logging

```python
common_utils.start_comet_ml_logging("michaelliang-dev")
```