WordNet Chatbot (wordnet_chatbot).


Summary of files:


1. File "wn_chatbot.py" launches the interactive "WordNet Chatbot". It has the following dependencies:

   a. intent_shared.py;

   b. wn_repository.py;

   c. data_pkl_xz_wordnet/lexicon_oewn.xz (INPUT);

   d. models/output_intents_params/*.h5 (INPUT);

   e. models/output_intents_schemas/*.json (INPUT);

   f. models/output_lookup_params/*.h5 (INPUT);

   g. models/output_lookup_schemas/*.json (INPUT).

2. File "wn_repository.py" generates compressed "pickle" repository of the WordNet data for the WordNet Chatbot content. It has the following dependencies:

   a. data_xml_wordnet/wn.xml (INPUT);

   b. data_pkl_xz_wordnet/lexicon_oewn.xz (OUTPUT).

3. File "intent_trainer.py" trains intent-based classification TensorFlow models with word embedding vectors. It has the following dependencies:

   a. intent_shared.py

   b. intent_models.py

   c. models_input_embeddings/glove.6B.100d.txt (OPTIONAL INPUT)

   d. models/input_configs/*.ini (INPUT);

   e. models/input_data/*.json (INPUT);

   f. models/output_intents_params/*.h5 (OUTPUT);

   g. models/output_intents_schemas/*.json (OUTPUT);

   h. models/output_lookup_params/*.h5 (OUTPUT);

   i. models/output_lookup_schemas/*.json (OUTPUT).

4. File "intent_tester.py" runs unit tests on the intent-based classification TensorFlow models with word embedding vectors. It has the following dependencies:

   a. intent_shared.py

   b. models/output_intents_params/*.h5 (INPUT);

   c. models/output_intents_schemas/*.json (INPUT);

   d. models/output_lookup_params/*.h5 (INPUT);

   e. models/output_lookup_schemas/*.json (INPUT).


Python Packages Requirements:


import sys; print("Python " + sys.version)

import os; print(os.__name__)

import sys; print(sys.__name__)

import random; print(random.__name__)

import configparser; print(configparser.__name__)

import string; print(string.__name__)

import xml; print(xml.__name__)

import pickle; print(pickle.__name__)

import lzma; print(lzma.__name__)

import heapq; print(heapq.__name__)

import datetime; print(datetime.__name__)

import json; print(json.__name__ + " " + json.__version__)

import matplotlib; print(matplotlib.__name__ + " " + matplotlib.__version__)

import sklearn; print(sklearn.__name__ + " " + sklearn.__version__)

import tensorflow; print(tensorflow.__name__ + " " + tensorflow.__version__)

import numpy; print(numpy.__name__ + " " + numpy.__version__)

import re; print(re.__name__ + " " + re.__version__)

import nltk; print(nltk.__name__ + " " + nltk.__version__)

import colorama; print(colorama.__name__ + " " + colorama.__version__)

import rapidfuzz; print(rapidfuzz.__name__ + " " + rapidfuzz.__version__)


Python 3.10.8 (tags/v3.10.8:aaaf517, Oct 11 2022, 16:50:30) [MSC v.1933 64 bit (AMD64)]

os

sys

random

configparser

string

xml

pickle

lzma

heapq

datetime

json 2.0.9

matplotlib 3.9.4

sklearn 1.5.2

tensorflow 2.8.3

numpy 1.26.4

re 2.2.1

nltk 3.7

colorama 0.4.6

rapidfuzz 3.13.0
