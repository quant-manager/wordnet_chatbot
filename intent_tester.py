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

###############################################################################

import os
import json
import configparser
from tensorflow.keras.models import model_from_json
from tensorflow.math import argmax
import tensorflow as tf
from intent_shared import \
    clean_intent_features_in_list, \
    ask_for_context_name, \
    DIR_PATH_MODELS_INPUT_CONFIGS, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS


###############################################################################

# Keep sentences short!
bool_print_misclassifications_only = True

str_current_context_name = ask_for_context_name()

with open(os.path.join(
        DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS,
        str_current_context_name + '_model_intents_schema.json')
        ) as file_handle :
    model = model_from_json(json.load(file_handle))
model.load_weights(os.path.join(
    DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS,
    str_current_context_name + '_model_intents_params.h5'))

with open(os.path.join(
        DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS,
        str_current_context_name + '_model_lookup_schema.json')
        ) as file_handle :
    model_string_inverse_lookup = model_from_json(json.load(file_handle))
model_string_inverse_lookup.load_weights(os.path.join(
    DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS,
    str_current_context_name + '_model_lookup_params.h5'))

###############################################################################

if str_current_context_name == "context_01_parts_of_speech" :
    dict_str_feature_to_str_true_intent_label = {
        "I want to learn about a noun, please." : "n", # noun
        "I am certainly interested in nouns a lot!" : "n", # noun
        "I wish I could learn a few adjectives, if it is possible at all." : "a", # adjective
        "If you happen to know some adjectives, I would be certainly happy to learn a few of these adjectives." : "a", # adjective
        "OK! Tell me about adverbs by example, please!" : "r", # adverb
        "Do you happen to know an adverb or two?" : "r", # adverb
        "If you really know some adjective satellites, then I would be delighted to know some of them." : "s", # adjective satellite
        "I can hardly believe that you are aware of so called adjective satellites. Tell me one of these adjective satellites." : "s", # adjective satellite
        "I am sure that you do not know what a pronoun is!" : "p", # pronoun
        "I need to know a pronoun!" : "p", # pronoun
        }
elif str_current_context_name == "context_02_sense_relation_type" :
    dict_str_feature_to_str_true_intent_label = {
        "I want a relation where derivation is involved." : "derivation",
        "May I know what pertains to this concept?" : "pertainym",
        "What is the antonym for this word?" : "antonym",
        "Tell me an antonym." : "antonym",
        "Do you have any idea about a more generalized classification." : "exemplifies",
        "What are the generalizing classes of this word?" : "exemplifies",
        "Could you give me an example of this concept?" : "is_exemplified_by",
        "What can be loosely related to this word?" : "also",
        "I need a verb for this participial adjective." : "participle",
        "Tell me anything similar to this concept." : "similar",
        "Could you give me an event?" : "other-event",
        "I want to know an agent." : "other-agent",
        "Do you have any result for to look at?" : "other-result",
        "Under the aegis of term?" : "other-by_means_of",
        "Undergoer." : "other-undergoer",
        "Instrument." : "other-instrument",
        "What is this term utilizing?" : "other-uses",
        "State." : "other-state",
        "Property." : "other-property",
        "Location." : "other-location",
        "Vehicle." : "other-vehicle",
        "Material." : "other-material",
        "Body part." : "other-body_part",
        "Destination." : "other-destination",
        }
elif str_current_context_name == "context_03_synset_relation_type" :
    dict_str_feature_to_str_true_intent_label = {
        "Tell me a more narrow subtype of another concept." : "hyponym",
        "I want to know something broader than that!!!" : "hypernym",
        "May I have something similar with what I see?" : "similar",
        "What can comprise a homogenious collection of these concepts?" : "holo_member",
        "I wish I knew what kind of homogenious elements comprise this concept." : "mero_member",
        "I am curious: what kind of heterogenious collections can contain this concept?" : "holo_part",
        "What are the heterogenious constituents of this thing?" : "mero_part",
        "What is an individual instance of this concept?" : "instance_hyponym",
        "What is its class or type?" : "instance_hypernym",
        "What is the scientific domain for this concept?" : "has_domain_topic",
        "What is the scientific category for it?" : "domain_topic",
        "What can be loosely related to this word?" : "also",
        "What are the generalizations?" : "exemplifies",
        "Could you give me an instance of this concept?" : "is_exemplified_by",
        "What is the cultural region for this concept?" : "has_domain_region",
        "What are the geographical feature for this concept?" : "domain_region",
        "What are the attributes of this concept?" : "attribute",
        "Tell me about the materials." : "mero_substance",
        "What is the products of it?" : "holo_substance",
        "What is it necessitated by?" : "is_entailed_by",
        "What does this concept entail among other concepts?" : "entails",
        "What was it resulted from?" : "is_caused_by",
        "What does it imply?" : "causes",
        }
elif str_current_context_name == "context_04_synset_lexicographer_noun_file" :
    dict_str_feature_to_str_true_intent_label = {
        "I need a man-made artifact." : "noun.artifact",
        "How about definition of some people" : "noun.person",
        "Tell me a plant noun!" : "noun.plant",
        "Give me an animal-related noun!" : "noun.animal",
        "I want to see some action!" : "noun.act",
        "Can you share some nouns about communication?" : "noun.communication",
        "I need nouns denoting a state." : "noun.state",
        "Tell me some nouns defining attributes for something." : "noun.attribute",
        "Do you have any location-related noun for me?" : "noun.location",
        "Please, provide nouns denoting cognition." : "noun.cognition",
        "Some substance-related word is something that I need." : "noun.substance",
        "I wish I had nouns denoting group." : "noun.group",
        "Tell me a few nouns denoting food." : "noun.food",
        "I am really interested in words about body parts!" : "noun.body",
        "How about a few natural objects for my consideration?" : "noun.object",
        "I want to learn a noun about quantities or metrics." : "noun.quantity",
        "How about something related to possession?" : "noun.possession",
        "I simply want to know some event from the world." : "noun.event",
        "I would like to know something about time!" : "noun.time",
        "Do you have any process from the real world for me?" : "noun.process",
        "A phenomenon is the topic that I want to learn now." : "noun.phenomenon",
        "I want to know about relations." : "noun.relation",
        "I am interested in feelings." : "noun.feeling",
        "I want to know nouns denoting shapes." : "noun.shape",
        "Provide unique beginner for nouns." : "noun.Tops",
        "I wish I knew a motive or two." : "noun.motive",
        }
elif str_current_context_name == "context_05_synset_lexicographer_adj_file" :
    dict_str_feature_to_str_true_intent_label = {
        "I need every adjective" : "adj.all",
        "May I have relational adjectives?" : "adj.pert",
        "Please, list pertainyms." : "adj.pert",
        "Participial adjectives are in my scope." : "adj.ppl",
        }
elif str_current_context_name == "context_06_synset_lexicographer_verb_file" :
    dict_str_feature_to_str_true_intent_label = {
        # "verbs of size, temperature change, intensifying, etc." : "verb.change",
        "Tell me a verb about change." : "verb.change",
        # "verbs of touching, hitting, tying, digging" : "verb.contact",
        "Tell me a verb about contact." : "verb.contact",
        # "verbs of telling, asking, ordering, singing" : "verb.communication",
        "Tell me a verb about communication." : "verb.communication",
        # "verbs of walking, flying, swimming" : "verb.motion",
        "Tell me a verb about motion." : "verb.motion",
        # "verbs of political and social activities and events" : "verb.social",
        "Tell me a verb about social." : "verb.social",
        # "verbs of buying, selling, owning" : "verb.possession",
        "Tell me a verb about possession." : "verb.possession",
        # "verbs of being, having, spatial relations" : "verb.stative",
        "Tell me a verb about stative." : "verb.stative",
        # "verbs of sewing, baking, painting, performing" : "verb.creation",
        "Tell me a verb about creation." : "verb.creation",
        # "verbs of thinking, judging, analyzing, doubting" : "verb.cognition",
        "Tell me a verb about cognition." : "verb.cognition",
        # "verbs of grooming, dressing and bodily care" : "verb.body",
        "Tell me a verb about body." : "verb.body",
        # "verbs of seeing, hearing, feeling" : "verb.perception",
        "Tell me a verb about perception." : "verb.perception",
        # "verbs of fighting, athletic activities" : "verb.competition",
        "Tell me a verb about competition." : "verb.competition",
        # "verbs of feeling" : "verb.emotion",
        "Tell me a verb about emotion." : "verb.emotion",
        # "verbs of eating and drinking" : "verb.consumption",
        "Tell me a verb about consumption." : "verb.consumption",
        # "verbs of raining, snowing, thawing, thundering" : "verb.weather",
        "Tell me a verb about weather." : "verb.weather",
        }
elif str_current_context_name == "context_07_sense_relation_syntactic_behaviour" :
    dict_str_feature_to_str_true_intent_label = {
        # ...
        }
elif str_current_context_name == "context_08_yes_no" :
    dict_str_feature_to_str_true_intent_label = {
        "yes" : "yes",
        "okey-dokey" : "yes",
        "alright" : "yes",
        "yea" : "yes",
        "in the affirmative" : "yes",
        "you bet" : "yes",
        "you may be sure" : "yes",
        "no" : "no",
        "no indeed" : "no",
        "absolutely not" : "no",
        "most certainly not" : "no",
        "of course not" : "no",
        "under no circumstances" : "no",
        "by no means" : "no",
        "not at all" : "no",
        "negative" : "no",
        "never" : "no",
        "not really" : "no",
        "no thanks" : "no",
        "nae" : "no",
        "nope" : "no",
        "nah" : "no",
        }
elif str_current_context_name == "context_09_positive_integers" :
    dict_str_feature_to_str_true_intent_label = {
        "1" : "1",
        "one" : "1",
        "5" : "5",
        "five" : "5",
        "10" : "10",
        "ten" : "10",
        "18" : "18",
        "eighteen" : "18",
        "22" : "22",
        "twenty two" : "22",
        "25" : "25",
        "twenty five" : "25",
        "one hundred" : "100",
        }
else :
    dict_str_feature_to_str_true_intent_label = None

(lst_str_testing_orig_intent_features, str_testing_true_intent_label) = zip(
    *dict_str_feature_to_str_true_intent_label.items())
config = configparser.ConfigParser()
config.read(os.path.join(
    DIR_PATH_MODELS_INPUT_CONFIGS, str_current_context_name + '.ini'))
conf_prep = config["intent.features.preprocessing"]
(lst_str_testing_clean_intent_features, set_all_features_dictionary) = \
    clean_intent_features_in_list(
        lst_str_intent_features = lst_str_testing_orig_intent_features,
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

###############################################################################

for (str_testing_orig_intent_feature, str_testing_clean_intent_feature,
     str_testing_true_intent_label) in \
     zip(lst_str_testing_orig_intent_features, lst_str_testing_clean_intent_features,
         str_testing_true_intent_label) :
    np_arr_flt_testing_predicted_intent_probs = model.predict(
        tf.constant([str_testing_clean_intent_feature]))

    np_arr_int_testing_predicted_intent_index = argmax(
        np_arr_flt_testing_predicted_intent_probs,
        axis = -1).numpy()
    np_arr_str_testing_predicted_intent_label = \
        model_string_inverse_lookup.predict(
            np_arr_int_testing_predicted_intent_index)
    str_testing_predicted_intent_label = \
        np_arr_str_testing_predicted_intent_label[0].decode('ascii')

    np_arr_int_testing_predicted_intent_indices = \
        tf.argsort(-np_arr_flt_testing_predicted_intent_probs,
                   axis = -1).numpy()[0,:]
    np_arr_str_testing_predicted_intent_labels = \
        model_string_inverse_lookup.predict(
            np_arr_int_testing_predicted_intent_indices)
    lst_str_testing_predicted_intent_labels = [
        bin_label.decode('ascii') for bin_label in
        np_arr_str_testing_predicted_intent_labels]

    if not bool_print_misclassifications_only or \
       str_testing_true_intent_label != str_testing_predicted_intent_label :
        print("Original Feature: " + str_testing_orig_intent_feature)
        print("Clean Feature: " + str_testing_clean_intent_feature)
        print("True Label: " + str_testing_true_intent_label)
        # print("Level 1 Predicted Label: " + str_testing_predicted_intent_label)
        print("Level 1 Predicted Label: " +
              lst_str_testing_predicted_intent_labels[0] +
              " (" + str(np_arr_flt_testing_predicted_intent_probs[0, np_arr_int_testing_predicted_intent_indices[0]]) + ")")
        print("Level 2 Predicted Label: " +
              lst_str_testing_predicted_intent_labels[1] +
              " (" + str(np_arr_flt_testing_predicted_intent_probs[0, np_arr_int_testing_predicted_intent_indices[1]]) + ")")
        print("Level 3 Predicted Label: " +
              lst_str_testing_predicted_intent_labels[2] +
              " (" + str(np_arr_flt_testing_predicted_intent_probs[0, np_arr_int_testing_predicted_intent_indices[2]]) + ")")
        print()

###############################################################################
