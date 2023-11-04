# Databricks notebook source
# MAGIC %md
# MAGIC ## Requirements

# COMMAND ----------

!pip install tortoise-tts

# COMMAND ----------

!pip install torch

# COMMAND ----------

!pip install torchvision

# COMMAND ----------

!pip install torchaudio

# COMMAND ----------

#!git clone https://github.com/neonbjb/tortoise-tts.git

# COMMAND ----------

#!cd tortoise-tts

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!python tortoise/do_tts.py --output_path ./res --text "Hi bros, how are you? I'm your solo guy that will bring you in the world of Artificial Intelligence!" --voice angie --preset fast

# COMMAND ----------

#https://github.com/neonbjb/tortoise-tts/blob/main/tortoise_tts.ipynb

# COMMAND ----------

# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# COMMAND ----------

tts = TextToSpeech(use_deepspeed=False, kv_cache=True)

# COMMAND ----------

text = "Hi bros, how are you? I'm your solo guy that will bring you in the world of Artificial Intelligence!"
# Pick one of the voices from the output above
voice = 'angie'
preset = 'high_quality'

# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)


# COMMAND ----------

torchaudio.save(f"results/generated-{voice}.wav", gen.squeeze(0).cpu(), 24000)


# COMMAND ----------


