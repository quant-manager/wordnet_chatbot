#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import json
import random
import configparser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import metrics
import tensorflow as tf
from intent_shared import \
    clean_intent_features_in_list, \
    ask_for_context_name
from intent_shared import \
    DIR_PATH_MODELS_INPUT_CONFIGS, \
    DIR_PATH_MODELS_INPUT_DATA, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS
from intent_models import create_string_lookup_model
from intent_models import create_string_inverse_lookup_model
from intent_models import \
    create_model_1, \
    create_model_2, \
    create_model_3, \
    create_model_4, \
    create_model_5, \
    create_model_6

###############################################################################

str_current_context_name = ask_for_context_name()

with open(os.path.join(
        DIR_PATH_MODELS_INPUT_DATA, str_current_context_name + '.json')) as file_handle :
    dict_current_context_of_intents = json.load(file_handle)

lst_str_all_intent_features = []
lst_str_all_intent_labels = []
set_str_unique_intent_labels = set()
for dict_intent in dict_current_context_of_intents['intents'] :
    for str_feature in dict_intent['features'] :
        lst_str_all_intent_features.append(str_feature)
        lst_str_all_intent_labels.append(dict_intent['label'])
    set_str_unique_intent_labels.add(dict_intent['label'])

bool_use_validation_subset = False
flt_traning_subset_size_fraction = .8
int_validation_subset_split_random_seed = 12345

# https://www.geeksforgeeks.org/python-shuffle-two-lists-with-same-order/
int_all_intent_features_count = len(lst_str_all_intent_features)
# Get a shuffled list of indices
lst_int_shuffled_indices = random.sample(
    range(int_all_intent_features_count), int_all_intent_features_count)
lst_str_all_intent_features = [lst_str_all_intent_features[int_obs_index]
                               for int_obs_index in lst_int_shuffled_indices]
lst_str_all_intent_labels = [lst_str_all_intent_labels[int_obs_index]
                             for int_obs_index in lst_int_shuffled_indices]


lst_str_unique_intent_labels = list(set_str_unique_intent_labels)
# https://docs.python.org/3/library/configparser.html
config = configparser.ConfigParser()
config.read(os.path.join(
    DIR_PATH_MODELS_INPUT_CONFIGS, str_current_context_name + '.ini'))
conf_prep = config["intent.features.preprocessing"]

(lst_str_all_clean_intent_features, set_all_features_dictionary) = \
    clean_intent_features_in_list(
        lst_str_intent_features = lst_str_all_intent_features,
        bool_remove_punctuation =
            conf_prep["bool_remove_punctuation"].lower().strip() == "true",
        bool_remove_digits =
            conf_prep["bool_remove_digits"].lower().strip() == "true",
        bool_replace_removed_characters_with_spaces =
            conf_prep["bool_replace_removed_characters_with_spaces"].lower(
                ).strip() == "true",
        bool_to_lower =
            conf_prep["bool_to_lower"].lower().strip() == "true",
        bool_remove_stopwords = # may create empty features
            conf_prep["bool_remove_stopwords"].lower().strip() == "true",
        str_stemmer = # None, "porter", "snowball", "lancaster".
            conf_prep["str_stemmer"].lower().strip(),
        str_lemmatizer = # None, "wordnet"
            conf_prep["str_lemmatizer"].lower().strip(),
        )

if bool_use_validation_subset :
    (lst_str_training_clean_intent_features,
     lst_str_validation_clean_intent_features,
     lst_str_training_intent_labels,
     lst_str_validation_intent_labels) = \
        train_test_split(
            lst_str_all_clean_intent_features,
            lst_str_all_intent_labels,
            train_size = flt_traning_subset_size_fraction,
            random_state = int_validation_subset_split_random_seed,)
    tf_arr_str_validation_clean_intent_features = tf.constant(
        lst_str_validation_clean_intent_features)
else :
    lst_str_training_clean_intent_features = lst_str_all_intent_features
    lst_str_training_intent_labels = lst_str_all_intent_labels

tf_arr_str_all_clean_intent_features = tf.constant(
    lst_str_all_clean_intent_features)
tf_arr_str_training_clean_intent_features = tf.constant(
    lst_str_training_clean_intent_features)

###############################################################################

model_string_lookup = create_string_lookup_model(
    lst_str_vocabulary = lst_str_unique_intent_labels)
model_string_inverse_lookup = create_string_inverse_lookup_model(
    lst_str_vocabulary = lst_str_unique_intent_labels)
tf_arr_int_all_intent_labels = tf.constant(model_string_lookup.predict(
    tf.constant(lst_str_all_intent_labels)))
tf_arr_int_training_intent_labels = tf.constant(model_string_lookup.predict(
    tf.constant(lst_str_training_intent_labels)))
if bool_use_validation_subset :
    tf_arr_int_validation_intent_labels = tf.constant(model_string_lookup.predict(
        tf.constant(lst_str_validation_intent_labels)))
if False :
    print(tf_arr_int_training_intent_labels)
    print(model_string_inverse_lookup.predict(tf_arr_int_training_intent_labels))

###############################################################################
# Defining Model Parameters
# Parameters may be overwritten later, depending on the model archictecture.

dict_int_model_id_to_fn_model_factory = \
{
    1 : create_model_1,
    2 : create_model_2,
    3 : create_model_3,
    4 : create_model_4,
    5 : create_model_5,
    6 : create_model_6,
}

################################
# Parameters for creating model:
inf_model_id = 1 # 1, 2, 3, 4, 5, 6
# the number of distinct tokens in the dictionary
int_vocabulary_size = len(set_all_features_dictionary)
int_embedding_vector_dimension = 16
int_max_sequence_length = 30
int_intent_label_classes_count = len(set_str_unique_intent_labels)
bool_use_glove_emdeddings = False
if bool_use_glove_emdeddings :
    int_embedding_vector_dimension = 100

################################
# Parameters for compiling model:
flt_learning_rate = 0.001

################################
# Parameters for fitting model:
int_epochs_count = 10000 # or 1000
int_batch_size = 32 # If unspecified, it will default to 32.
bool_shuffle = True
bool_use_early_stopping = True
# Number of epochs with no improvement after which training will be stopped.
str_early_stopping_monitor = 'sparse_categorical_crossentropy'
# str_early_stopping_monitor = 'sparse_categorical_accuracy'
int_early_stopping_patience = 100
bool_early_stopping_restore_best_weights = True
flt_early_stopping_min_delta = .0001
str_early_stopping_mode = 'auto'

###############################################################################
# Model Creation

# https://community.deeplearning.ai/t/recommended-way-to-tokenize-new-code/517807
# https://www.tensorflow.org/guide/keras/preprocessing_layers
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
#
# Because we've passed the vocabulary directly, we don't need to adapt
# the layer - the vocabulary is already set. The vocabulary contains the
# padding token ('') and OOV token ('[UNK]')
# as well as the passed tokens.
text_vectorization_layer = TextVectorization(
    max_tokens = int_vocabulary_size + 2, # add "" and "<UNK>" tokens
    standardize = 'lower_and_strip_punctuation',
    split = 'whitespace',
    output_mode = 'int',
    output_sequence_length = int_max_sequence_length,
    pad_to_max_tokens = False,
    vocabulary = list(set_all_features_dictionary),
    )
if len(text_vectorization_layer.get_vocabulary()) <= 2 : # ['', '[UNK]']
    text_vectorization_layer.adapt(tf_arr_str_all_clean_intent_features)
if False :
    print('Vocabulary:', text_vectorization_layer.get_vocabulary(), '\n')
    tensor_vectorized_text = text_vectorization_layer(
        tf_arr_str_all_clean_intent_features)
    print("Original vectorized corpus:")
    print(tensor_vectorized_text)

fn_creat_model = dict_int_model_id_to_fn_model_factory[inf_model_id]
if inf_model_id == 3 :
    bool_use_early_stopping = False
elif inf_model_id == 4 :
    #bool_use_early_stopping = False
    flt_learning_rate = 0.0001
elif inf_model_id == 5 :
    # Model 5 overwritten default paramaters' values with customized values.
    bool_use_early_stopping = False
    # str_early_stopping_monitor = 'loss'
    # int_early_stopping_patience = 400
    int_epochs_count = 5000
model = fn_creat_model(
    text_vectorization_layer = text_vectorization_layer,
    int_embedding_vector_dimension = int_embedding_vector_dimension,
    int_max_sequence_length = int_max_sequence_length,
    int_intent_label_classes_count = int_intent_label_classes_count,
    bool_use_glove_emdeddings = bool_use_glove_emdeddings,)

###############################################################################
# Model Compiling

model.compile(
    loss = SparseCategoricalCrossentropy(),
    optimizer = Adam(learning_rate = flt_learning_rate),
    metrics = [metrics.SparseCategoricalCrossentropy(),
               metrics.SparseCategoricalAccuracy()],)
if False :
    print(model.summary())

###############################################################################
# Model Fitting

lst_callbacks = []
if bool_use_early_stopping :
    callback_early_stoping = tf.keras.callbacks.EarlyStopping(
        monitor = str_early_stopping_monitor,
        patience = int_early_stopping_patience,
        restore_best_weights = bool_early_stopping_restore_best_weights,
        min_delta = flt_early_stopping_min_delta,
        mode = str_early_stopping_mode,
        )
    lst_callbacks.append(callback_early_stoping)

if bool_use_validation_subset :
    tpl_validation_data_features_labels = (
        tf_arr_str_validation_clean_intent_features,
        tf_arr_int_validation_intent_labels)
else :
    tpl_validation_data_features_labels = None

history_of_training = model.fit(
    tf_arr_str_training_clean_intent_features,
    tf_arr_int_training_intent_labels,
    epochs = int_epochs_count,
    batch_size = int_batch_size,
    shuffle = bool_shuffle,
    validation_data = tpl_validation_data_features_labels,
    callbacks = lst_callbacks,
    )

###############################################################################
# Model Diagnostics

def generate_fitting_diagnostics_plot(
        history_of_training, str_statistic_name_1, str_statistic_name_2 = None) :
    #mpl.style.use('ggplot') # mpl.style.available
    plt.figure(figsize = (11., 8.5))
    plt.plot(history_of_training.history[str_statistic_name_1])
    if str_statistic_name_2 is not None :
        plt.plot(history_of_training.history[str_statistic_name_2])
    plt.title('Model ' + str_statistic_name_1)
    plt.ylabel(str_statistic_name_1)
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

generate_fitting_diagnostics_plot(
    history_of_training,
    'sparse_categorical_accuracy',
    'val_sparse_categorical_accuracy' if bool_use_validation_subset else None,
    )
generate_fitting_diagnostics_plot(
    history_of_training,
    'loss',
    'val_loss' if bool_use_validation_subset else None,
    # 'sparse_categorical_crossentropy',
    # 'val_sparse_categorical_crossentropy' if bool_use_validation_subset else None,
    )

###############################################################################
# Model Evaluation

print()
print("Model Evaluation for the training data subset of size {:d}:".format(
        tf_arr_str_training_clean_intent_features.shape[0]))
print(model.evaluate(
    tf_arr_str_training_clean_intent_features,
    tf_arr_int_training_intent_labels,
    batch_size = int_batch_size))
if bool_use_validation_subset :
    print()
    print("Model Evaluation for the validation data subset of size {:d}:".format(
        tf_arr_str_validation_clean_intent_features.shape[0]))
    print(model.evaluate(
        tf_arr_str_validation_clean_intent_features,
        tf_arr_int_validation_intent_labels,
        batch_size = int_batch_size))

###############################################################################

# https://www.tensorflow.org/guide/keras/serialization_and_saving
# https://github.com/keras-team/tf-keras/issues/574
model.save_weights(os.path.join(
    DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS,
    str_current_context_name + '_model_intents_params.h5'))
with open(os.path.join(
        DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS,
        str_current_context_name + '_model_intents_schema.json'),
        "wt") as file_handle :
    json.dump(model.to_json(), file_handle)

model_string_inverse_lookup.save_weights(os.path.join(
    DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS,
    str_current_context_name + '_model_lookup_params.h5'))
with open(os.path.join(
        DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS,
        str_current_context_name + '_model_lookup_schema.json'),
        "wt") as file_handle :
    json.dump(model_string_inverse_lookup.to_json(), file_handle)
