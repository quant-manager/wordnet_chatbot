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

import string
import re
import os
# https://towardsai.net/p/l/stemming-porter-vs-snowball-vs-lancaster
# https://www.nltk.org/api/nltk.html
# https://www.nltk.org/api/nltk.stem.porter.html#nltk.stem.porter.PorterStemmer
from nltk.stem.porter import PorterStemmer
# https://www.nltk.org/api/nltk.stem.snowball.html#nltk.stem.snowball.SnowballStemmer
from nltk.stem.snowball import SnowballStemmer
# https://www.nltk.org/api/nltk.stem.lancaster.html#nltk.stem.lancaster.LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer
# https://www.nltk.org/api/nltk.stem.wordnet.html#nltk.stem.wordnet.WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

lst_str_contexts_names = [
    "context_01_parts_of_speech",
    "context_02_sense_relation_type",
    "context_03_synset_relation_type",
    "context_04_synset_lexicographer_noun_file",
    "context_05_synset_lexicographer_adj_file",
    "context_06_synset_lexicographer_verb_file",
    # "context_07_sense_relation_syntactic_behaviour",
    "context_08_yes_no",
    "context_09_positive_integers",
    ]

DIR_PATH_MODELS_INPUT_CONFIGS          = os.path.join("models", "input_configs")
DIR_PATH_MODELS_INPUT_DATA             = os.path.join("models", "input_data")
DIR_PATH_MODELS_INPUT_EMBEDDINGS       = os.path.join("models_input_embeddings")
DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS  = os.path.join("models", "output_intents_params")
DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS = os.path.join("models", "output_intents_schemas")
DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS   = os.path.join("models", "output_lookup_params")
DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS  = os.path.join("models", "output_lookup_schemas")


def ask_for_context_name() :
    global lst_str_contexts_names
    int_context_count = len(lst_str_contexts_names)
    bool_is_context_selected = False
    str_current_context_name = None
    while not bool_is_context_selected :
        print("The list of context models:\n")
        for (i, str_context_name) in enumerate(lst_str_contexts_names) :
            print("{0}. {1}.".format(i + 1, str_context_name))
        print("\nChoose the index of the context model:")
        str_context_index = input().strip()
        if str_context_index.isdigit() :
            int_context_index = int(str_context_index)
            if 1 <= int_context_index <= int_context_count :
                str_current_context_name = lst_str_contexts_names[int_context_index - 1]
                bool_is_context_selected = True
            else :
                print("Error: the index must be between 1 and {0}.".format(int_context_count))
        else :
            print("Error: the index must be a positive integer from 1 to {0}.".format(int_context_count))
    return str_current_context_name


def clean_intent_feature(
        str_feature,
        str_chars_to_remove = "",
        bool_replace_removed_characters_with_spaces = True,
        bool_to_lower = True,
        set_str_stopwords = set(),
        stemmer = None, # None, "PorterStemmer", "SnowballStemmer", "LancasterStemmer".
        lemmatizer = None, # None, "WordNetLemmatizer"
        ) :

    str_upd_feature = str_feature

    # str_upd_feature = re.sub(r'[^a-zA-z.?!\']', ' ', str_upd_feature)
    # str_upd_feature = re.sub(r'[ ]+', ' ', str_upd_feature)

    if bool_replace_removed_characters_with_spaces :
        str_upd_feature = re.sub('[' + re.escape(str_chars_to_remove) + ']',
                                   ' ', str_feature)
        # str_upd_feature = str_feature.translate(str.maketrans(
        #     str_chars_to_remove,' '*len(str_chars_to_remove),'')) # alternative
    else :
        str_upd_feature = re.sub('[' + re.escape(str_chars_to_remove) + ']',
                                   '', str_feature)
        # str_upd_feature = str_feature.translate(str.maketrans(
        #     '','',str_chars_to_remove)) # alternative

    if bool_to_lower :
        str_upd_feature = str_upd_feature.lower()

    lst_str_upd_feature = str_upd_feature.split(sep = " ")
    if stemmer is not None :
        if lemmatizer is not None :
            str_upd_feature = ' '.join(
                [stemmer.stem(lemmatizer.lemmatize(str_word))
                 for str_word in lst_str_upd_feature
                 if str_word not in set_str_stopwords])
        else :
            str_upd_feature = ' '.join(
                [stemmer.stem(str_word)
                 for str_word in lst_str_upd_feature
                 if str_word not in set_str_stopwords])
    else :
        if lemmatizer is not None :
            str_upd_feature = ' '.join(
                [lemmatizer.lemmatize(str_word)
                 for str_word in lst_str_upd_feature
                 if str_word not in set_str_stopwords])
        else :
            str_upd_feature = ' '.join(
                [str_word
                 for str_word in lst_str_upd_feature
                 if str_word not in set_str_stopwords])

    str_upd_feature = str_upd_feature.strip()

    return str_upd_feature


def clean_intent_features_in_dict(
        dict_str_intent_features_to_str_labels,
        bool_remove_punctuation = True,
        bool_remove_digits = True,
        bool_replace_removed_characters_with_spaces = True,
        bool_to_lower = True,
        bool_remove_stopwords = False,
        str_stemmer = None, # None, "porter", "snowball", "lancaster".
        str_lemmatizer = None, # None, "wordnet"
        ) :

    str_chars_to_remove = string.punctuation if bool_remove_punctuation else ""
    if bool_remove_digits :
        str_chars_to_remove += "0123456789"

    if bool_remove_stopwords :
        if bool_remove_punctuation :
            set_str_stopwords = set(stopwords.words('english') +
                                    list(str_chars_to_remove))
        else :
            set_str_stopwords = set(stopwords.words('english'))
    else :
        set_str_stopwords = set()

    if str_stemmer is None :
        stemmer = None
    elif str_stemmer == "porter" :
        stemmer = PorterStemmer()
    elif str_stemmer == "snowball" :
        stemmer = SnowballStemmer(language = "english", ignore_stopwords = True)
    elif str_stemmer == "lancaster" :
        stemmer = LancasterStemmer(rule_tuple = None, strip_prefix_flag = False)
    else :
        stemmer = None

    if str_lemmatizer is None :
        lemmatizer = None
    elif str_lemmatizer == "wordnet" :
        lemmatizer = WordNetLemmatizer()
    else :
        lemmatizer = None

    dict_str_clean_intent_features_to_str_labels = {}
    set_all_features_dictionary = set()
    for (str_original_intent_feature, str_label) in \
         dict_str_intent_features_to_str_labels.items() :
        str_clean_intent_feature = clean_intent_feature(
            str_feature = str_original_intent_feature,
            str_chars_to_remove = str_chars_to_remove,
            bool_replace_removed_characters_with_spaces = \
                bool_replace_removed_characters_with_spaces,
            bool_to_lower = bool_to_lower,
            set_str_stopwords = set_str_stopwords,
            stemmer = stemmer,
            lemmatizer = lemmatizer,)
        set_feature_dictionary = set(str_clean_intent_feature.split())
        set_all_features_dictionary = set_all_features_dictionary.union(
            set_feature_dictionary)

        dict_str_clean_intent_features_to_str_labels[
            str_clean_intent_feature] = str_label
    return (dict_str_clean_intent_features_to_str_labels,
            set_all_features_dictionary)


def clean_intent_features_in_list(
        lst_str_intent_features,
        bool_remove_punctuation = True,
        bool_remove_digits = True,
        bool_replace_removed_characters_with_spaces = True,
        bool_to_lower = True,
        bool_remove_stopwords = False,
        str_stemmer = None, # None, "porter", "snowball", "lancaster".
        str_lemmatizer = None, # None, "wordnet".
        ) :

    str_chars_to_remove = string.punctuation if bool_remove_punctuation else ""
    if bool_remove_digits :
        str_chars_to_remove += "0123456789"

    if bool_remove_stopwords :
        if bool_remove_punctuation :
            set_str_stopwords = set(stopwords.words('english') +
                                    list(str_chars_to_remove))
        else :
            set_str_stopwords = set(stopwords.words('english'))
    else :
        set_str_stopwords = set()

    if str_stemmer is None :
        stemmer = None
    elif str_stemmer == "porter" :
        stemmer = PorterStemmer()
    elif str_stemmer == "snowball" :
        stemmer = SnowballStemmer(language = "english", ignore_stopwords = True)
    elif str_stemmer == "lancaster" :
        stemmer = LancasterStemmer(rule_tuple = None, strip_prefix_flag = False)
    else :
        stemmer = None

    if str_lemmatizer is None :
        lemmatizer = None
    elif str_lemmatizer == "wordnet" :
        lemmatizer = WordNetLemmatizer()
    else :
        lemmatizer = None

    lst_str_clean_intent_features = [None] * len(lst_str_intent_features)
    set_all_features_dictionary = set()
    for int_intent_feature_index in range(len(lst_str_intent_features)) :
        str_original_intent_feature = lst_str_intent_features[
            int_intent_feature_index]
        str_clean_intent_feature = \
            clean_intent_feature(
                str_feature = str_original_intent_feature,
                str_chars_to_remove = str_chars_to_remove,
                bool_replace_removed_characters_with_spaces = \
                    bool_replace_removed_characters_with_spaces,
                bool_to_lower = bool_to_lower,
                set_str_stopwords = set_str_stopwords,
                stemmer = stemmer,
                lemmatizer = lemmatizer,)
        set_feature_dictionary = set(str_clean_intent_feature.split())
        set_all_features_dictionary = set_all_features_dictionary.union(
            set_feature_dictionary)

        lst_str_clean_intent_features[int_intent_feature_index] = \
            str_clean_intent_feature

    return (lst_str_clean_intent_features, set_all_features_dictionary)

