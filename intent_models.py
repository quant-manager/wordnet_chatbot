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
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import \
    Dense, \
    Embedding, \
    GlobalAveragePooling1D, \
    LSTM, \
    Dropout, \
    Flatten, \
    InputLayer, \
    StringLookup, \
    Bidirectional, \
    Conv1D, \
    Input #, \
    # TextVectorization
import tensorflow as tf
from intent_shared import DIR_PATH_MODELS_INPUT_EMBEDDINGS


def generate_glove_emdeddings(
        text_vectorization_layer, int_embedding_vector_dimension) :
    '''
    https://zenodo.org/records/4925376
    https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation
    https://nlp.stanford.edu/projects/glove/
    Wikipedia 2014 + Gigaword 5
    (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors,
     822 MB download): glove.6B.zip
    '''
    # !wget https://nlp.stanford.edu/data/glove.6B.zip
    str_glove_file_name = "glove.6B.100d.txt"
    int_vocabulary_size = len(text_vectorization_layer.get_vocabulary())
    if str_glove_file_name == "glove.6B.100d.txt" and \
       int_embedding_vector_dimension != 100 :
        raise ValueError(
            'Embedding vector dimension must be 100 for file {:s}'.format(
                str_glove_file_name))

    dict_str_word_to_np_arr_embedding_vector = {}
    with open(os.path.join(
            DIR_PATH_MODELS_INPUT_EMBEDDINGS, str_glove_file_name),
            encoding = "utf8" ) as file_handle :
        for str_file_line in file_handle:
            lst_str_line_tokens = str_file_line.split()
            str_word = lst_str_line_tokens[0]
            np_arr_emdedding_vector = np.asarray(
                lst_str_line_tokens[1:], dtype = 'float32')
            dict_str_word_to_np_arr_embedding_vector[str_word] = \
                np_arr_emdedding_vector

    np_arr_2d_all_embedding_vectors = np.stack(
        list(dict_str_word_to_np_arr_embedding_vector.values())) # 400000-by-100
    (flt_emb_mean, flt_emb_stddev) = (np_arr_2d_all_embedding_vectors.mean(),
                                      np_arr_2d_all_embedding_vectors.std())
    if False :
        print('Found {:d} word vectors.'.format(
            len(dict_str_word_to_np_arr_embedding_vector))) # 400000
        print(flt_emb_mean, flt_emb_stddev) # (0.004451992, 0.4081574)

    if True :
        np_arr_2d_vocab_embedding_vectors = np.random.normal(
            flt_emb_mean, flt_emb_stddev,
            (int_vocabulary_size, int_embedding_vector_dimension))
    else :
        np_arr_2d_vocab_embedding_vectors = np.zeros(
            (int_vocabulary_size, int_embedding_vector_dimension))
    lst_str_words = text_vectorization_layer.get_vocabulary()
    for (int_word_index, str_word) in enumerate(lst_str_words) :
        np_arr_emdedding_vector = dict_str_word_to_np_arr_embedding_vector.get(
            str_word)
        if np_arr_emdedding_vector is not None:
            np_arr_2d_vocab_embedding_vectors[int_word_index] = \
                np_arr_emdedding_vector
    return np_arr_2d_vocab_embedding_vectors


def create_string_lookup_model(lst_str_vocabulary) :
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
    # https://www.tensorflow.org/guide/keras/
    #   making_new_layers_and_models_via_subclassing
    # https://stackoverflow.com/questions/72252109/
    #   implementing-label-encoder-as-a-tensorflow-preprocessing-layer/72254671
    # https://github.com/keras-team/tf-keras/issues/77
    model_string_lookup = Sequential([
        InputLayer(input_shape = [], dtype = tf.string), # requred layer!
        StringLookup(
            vocabulary = lst_str_vocabulary,
            # The number of out-of-vocabulary tokens to use. If this value is 0,
            # OOV inputs will cause an error when calling the layer. Defaults to 1.
            num_oov_indices = 0, # the set of valid labels is known and fixed
            invert = False),
        ])
    return model_string_lookup


def create_string_inverse_lookup_model(lst_str_vocabulary) :
    model_string_inverse_lookup = Sequential([
        tf.keras.layers.InputLayer(input_shape=[], dtype=tf.int64),
        tf.keras.layers.StringLookup(
            vocabulary = lst_str_vocabulary,
            num_oov_indices = 0,  # the set of valid labels is known and fixed
            invert = True),
        ])
    return model_string_inverse_lookup


def create_model_1(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    model = Sequential([
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        GlobalAveragePooling1D(),
        Dense(units = 16, activation = "relu"),
        Dense(units = 16, activation = "relu"),
        Dense(units = int_intent_label_classes_count, activation = 'softmax'),
        ])
    return model


def create_model_2(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    model = Sequential([
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        # LSTM(units = 100, return_sequences = False),
        LSTM(units = 100, return_sequences = False, input_shape = (
            None, int_max_sequence_length, int_embedding_vector_dimension)),
        Dense(units = int_intent_label_classes_count, activation = 'softmax'),
        ])
    return model


def create_model_3(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    model = Sequential([
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        Flatten(),
        Dense(units = 128, activation='relu'),
        Dropout(rate = .5), # Float between 0 and 1. Fraction of the input units to drop.
        Dense(units = 64, activation='relu'),
        Dropout(rate = .5),
        Dense(units = int_intent_label_classes_count, activation = 'softmax',),
        ])
    return model


def create_model_4(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    int_lstm_units_count = 100
    model = Sequential([
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        Bidirectional(LSTM(
            units = int_lstm_units_count,
            dropout = .1, # or .2
            return_sequences = False,
            )), 
        Dense(units = int_lstm_units_count, activation = 'relu',),
        Dropout(rate = .4), # or .5
        Dense(units = int_intent_label_classes_count, activation = 'softmax',),
        ])
    return model


def create_model_5(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    model = Sequential([
        Input(shape = (), dtype = tf.string,), # optional
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        Conv1D(
            filters = 32,
            kernel_size = 5,
            activation = "relu",
            kernel_initializer = tf.keras.initializers.GlorotNormal(),
            bias_regularizer = tf.keras.regularizers.L2(0.0001),
            kernel_regularizer = tf.keras.regularizers.L2(0.0001),
            activity_regularizer = tf.keras.regularizers.L2(0.0001),),
        Dropout(rate = .3),
        LSTM(units = 32, dropout = .3, return_sequences = True,),
        LSTM(units = 16, dropout = .3, return_sequences = False,),
        Dense(
            units = 128,
            activation = "relu",
            activity_regularizer = tf.keras.regularizers.L2(0.0001),),
        Dropout(rate = .6),
        Dense(
            units = int_intent_label_classes_count,
            activation = 'softmax',
            activity_regularizer = tf.keras.regularizers.L2(0.0001),),
        ])
    # print(model.layers)

    return model


def create_model_6(
        text_vectorization_layer, int_embedding_vector_dimension,
        int_max_sequence_length, int_intent_label_classes_count,
        bool_use_glove_emdeddings = False) :
    # https://stackoverflow.com/questions/68844792/
    #  lstm-will-not-use-cudnn-kernels-since-it-doesnt-meet-the-criteria-it-will-use
    # WARNING:tensorflow:Layer lstm_4 will not use cuDNN kernels since it
    # doesn't meet the criteria. It will use a generic GPU kernel as fallback
    # when running on GPU.
    # Converges very well, but slowly as measured by fitting time.

    if bool_use_glove_emdeddings :
        bool_is_embedding_layer_trainable = False
        np_arr_2d_vocab_embedding_vectors = generate_glove_emdeddings(
            text_vectorization_layer = text_vectorization_layer,
            int_embedding_vector_dimension = int_embedding_vector_dimension,)
        lst_np_arr_2d_vocab_embedding_weights = [
            np_arr_2d_vocab_embedding_vectors]
    else :
        bool_is_embedding_layer_trainable = True
        lst_np_arr_2d_vocab_embedding_weights = None

    model = Sequential([
        text_vectorization_layer,
        Embedding(
            input_dim = len(text_vectorization_layer.get_vocabulary()),
            output_dim = int_embedding_vector_dimension,
            input_length = int_max_sequence_length,
            mask_zero = True,
            trainable = bool_is_embedding_layer_trainable,
            weights = lst_np_arr_2d_vocab_embedding_weights,
            ),
        Bidirectional(LSTM(
            units = 256,
            return_sequences = True,
            recurrent_dropout = 0.1,
            dropout = 0.1,
            #recurrent_dropout = 0.0, # Use to avoid the above WARNING
            #dropout = 0.0, # Use to avoid the above WARNING
            ), 'concat'),
        Dropout(rate = 0.3),
        LSTM(
            units = 256,
            return_sequences = False,
            recurrent_dropout = 0.1,
            dropout = 0.1,
            #recurrent_dropout = 0.0, # Use to avoid the above WARNING
            #dropout = 0.0, # Use to avoid the above WARNING
        ),
        Dropout(rate = .3),
        Dense(units = 50, activation='relu'),
        Dropout(rate = .3),
        Dense(units = int_intent_label_classes_count, activation = 'softmax',),
        ])
    return model
