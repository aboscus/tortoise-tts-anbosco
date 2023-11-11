# Databricks notebook source
# MAGIC %md
# MAGIC Welcome to Tortoise! 🐢🐢🐢🐢
# MAGIC
# MAGIC Before you begin, I **strongly** recommend you turn on a GPU runtime.
# MAGIC
# MAGIC There's a reason this is called "Tortoise" - this model takes up to a minute to perform inference for a single sentence on a GPU. Expect waits on the order of hours on a CPU.

# COMMAND ----------

#first follow the instructions in the README.md file under Local Installation
!pip3 install -r requirements.txt
# !python3 setup.py install

# COMMAND ----------

# Imports used through the rest of the notebook.
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(use_deepspeed=False, kv_cache=True)

# COMMAND ----------

# Tortoise will attempt to mimic voices you provide. It comes pre-packaged
# with some voices you might recognize.

# Let's list all the voices available. These are just some random clips I've gathered
# from the internet as well as a few voices from the training dataset.
# Feel free to add your own clips to the voices/ folder.
%ls tortoise/voices

IPython.display.Audio('tortoise/voices/tom/1.wav')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Long text

# COMMAND ----------

# This is the text that will be spoken.

long_text = """

Hey there, tech enthusiasts! Welcome to another edition of your favorite tech dive, where we'll unravel the latest buzz from the world of technology. Let's kick things off with a Hollywood twist.

So, the Writers Guild of America and Hollywood studios have finally called a truce, ending the writers' strike. But what's caught our attention is the role of AI in this drama. The deal? No AI in the scriptwriting game. It's like the studios and writers said, "Hey, let's keep the creative magic human, okay?" But hold your horses, the actors' union, SAG-AFTRA, is still on strike, and they're eyeing a showdown with the video game industry. The drama never ends!


Now, shifting gears to the startup scene in San Fran, we've got Talkdesk, a customer service software company, making headlines for all the wrong reasons. Layoffs. Again. For the third time in 14 months. Word on the street is, they're cutting costs faster than you can say "customer service." Rumor has it, they've pink-slipped folks in February and August. We're left wondering, how many are still sipping lattes at Talkdesk?


Guess who's making it rain in the world of AI? OpenAI! Talks are buzzing that they're eyeing a deal to sell shares, and the numbers are mind-blowing — from $29 billion to potentially $90 billion! And guess what? Microsoft's got a cozy 49% stake in this AI extravaganza. With ChatGPT turning heads, OpenAI's predicting a cool $1 billion in revenue by 2023. Buckle up; the AI ride is getting wild!


Jumping across the pond, the European Union is throwing shade at generative AI tools, warning about the risks to free and fair debate. They're side-eyeing AI-generated disinformation, pointing fingers at none other than OpenAI. Looks like the EU wants everyone on board their anti-disinformation Code. OpenAI, you feeling the pressure?


San Francisco's startup Kolena just hit the jackpot with a $15 million funding round. They're all about testing, benchmarking, and validating the performance of AI models. Privacy-focused and ready to tackle risks, Kolena's saying no to the usual data uploads. Are they onto something big? Only time will tell.


OpenAI's back in the game, publishing a tech paper on GPT-4's image-analyzing tools. They're all about safeguards, preventing the AI from going rogue. But here's the catch — even with safeguards, GPT-4V sometimes throws a curveball, creating made-up terms like it's playing word salad. A little glitch in the matrix, huh?


Ever wished your TV could recommend shows based on your quirky questions? Tubi's got you covered with Rabbit AI, powered by OpenAI's GPT-4. Asking for shark comedies? Rabbit AI's got your back. It's like having your own AI-powered TV genie. Beta testing is on, and soon you'll be drowning in personalized recommendations.


Erudit, a startup in the employee sentiment game, just bagged $10 million in funding. Monitoring metrics from Slack to Zoom, they claim to anonymize chat data. But, and it's a big but, a 2021 survey says employees are stressing over surveillance. How do you balance metrics and trust? Erudit's got some explaining to do.


Let's hit the open-source dance floor with PartyKit, raking in $2.5 million in pre-seed funding. It's like the cool kid in town, offering libraries for multiplayer app fun. Think Vercel or Netlify, but with a dash of collaboration. Bring your code, integrate, and party on Cloudflare. Sounds like a tech bash we want an invite to!


Over in Taiwan, AI chipmaker Kneron's playing with some serious cash — $49 million in an extended Series B round. They're all about low-powered AI chips for your favorite driverless vehicles. With a customer base including Garmin and Naver, Kneron's revving up the AI engine. Watch out, Tesla!


Signal's president, Meredith Whittaker, just threw shade at AI, calling it a surveillance technology. She's saying it's like peanut butter and jelly with big data and targeting. But not all AIs are created equal, she claims. Signal's got its own small on-device model, giving a face blur feature. A privacy advocate in the house!


In the writing industry, AI is making waves, according to May Habib, CEO of Writer. Forget job losses; she says AI's creating gigs faster than we can type. But wait, not everyone's on board. The Writers Guild had a tiff, but they kissed and made up after a strike. Let the AI writing revolution begin!


Getty Images is trying its hand at generative AI art. They've got a tool that's "commercially safer" than the rivals. Using stock content, it creates images from text prompts. Getty's promising not to add these to their library. Plus, contributors get a slice of the pie. It's art without the legal drama.


Microsoft's making moves with Snapchat, inserting ads into My AI chatbot. Your casual dinner chat could now come with a sponsored link. Imagine AI recommending the hottest spots in town. It's like your AI friend just became your personal advertiser. Ads: the new dinner conversation.


OpenAI's ChatGPT is leveling up, now with voice and image features. It's like your favorite text buddy just learned some cool new tricks. Soon, you'll be chatting away and searching with just a snap. It's like the Swiss Army knife of AI assistants.


Elicit, a startup in the academic arena, brings a research assistant powered by language models. Imagine streamlining those tedious literature reviews. Over 200,000 users are saying, "Goodbye, sleepless nights." Elicit's got the academic world buzzing.


Amazon's throwing serious cash at AI startup Anthropic — up to $4 billion kind of serious. They're eyeing a "frontier model," ten times more potent than today's AI powerhouses. It's like Amazon just entered the AI arms race. Who's ready for the future?


And last but not least, Madrid-based Correcto just scored $7 million in seed funding. They're all about the Spanish language, correcting your written Spanish with Grammarly-style finesse. And here's the kicker — they tapped into OpenAI's API for a quasi-generative-writing feature. ¡Viva la corrección!


That's a wrap, tech-heads! We've taken you on a rollercoaster ride through the hottest tech stories. Until next time, keep your devices charged, your code bug-free, and your AI assistants well-behaved. Catch you on the flip side!


"""


# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "fast"

# COMMAND ----------

import nltk
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

# Download NLTK data for sentence tokenization
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Use NLTK to tokenize the text into sentences
sentences = sent_tokenize(long_text)

# Create a Spark session

# Create a DataFrame with sentences and their indices
data = [(index, sentence) for index, sentence in enumerate(sentences)]
df = spark.createDataFrame(data, ["index", "sentence"])

# Add a unique ID column to the DataFrame
df = df.withColumn("id", monotonically_increasing_id())

# Show the resulting DataFrame
df.show(truncate=False)
sentences = df.orderBy("index").select("sentence").rdd.flatMap(lambda x: x).collect()


# COMMAND ----------

from tortoise.utils.text import split_and_recombine_text
from time import time
import os

outpath = "results/longform/"
seed=1
voice = "tom"
preset = "fast"

voice_outpath = os.path.join(outpath, voice)
os.makedirs(voice_outpath, exist_ok=True)

voice_samples, conditioning_latents = load_voice(voice)

all_parts = []
for j, text in enumerate(sentences):
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                              preset=preset, k=1, use_deterministic_seed=seed)
    gen = gen.squeeze(0).cpu()
    torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
    all_parts.append(gen)

full_audio = torch.cat(all_parts, dim=-1)
torchaudio.save(os.path.join(voice_outpath, 'combined_podcast.wav'), full_audio, 24000)
IPython.display.Audio(os.path.join(voice_outpath, 'combined_podcast.wav'))

# COMMAND ----------


