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
import sys
import json
import heapq
from datetime import datetime
import configparser
from tensorflow.keras.models import model_from_json
# from tensorflow.math import argmax
import tensorflow as tf
import random
import colorama
from colorama import Fore, Style #, Back
# pip install --upgrade rapidfuzz
from rapidfuzz.distance import DamerauLevenshtein
from wn_repository import \
    Lexicon, \
    LexicalEntry, \
    Lemma, \
    Pronunciation, \
    Form, \
    Sense, \
    SyntacticBehaviour,\
    SenseRelation, \
    Synset, \
    SynsetRelation, \
    Definition, \
    Example, \
    dict_parts_of_speech_code_to_name, \
    dict_relations_between_senses_name_to_descr, \
    dict_other_relations_between_senses_name_to_descr, \
    dict_synset_lexicographer_noun_files_name_to_descr, \
    dict_synset_lexicographer_adj_files_name_to_descr, \
    dict_synset_lexicographer_verb_files_name_to_descr, \
    dict_sense_relations_syntactic_behaviours_code_to_descr, \
    dict_relations_between_synsets_name_to_descr, \
    DIR_PATH_DATA_PKL_XZ_WORDNET, \
    FILE_NAME_DATA_PKL_XZ_WORDNET
from intent_shared import \
    clean_intent_features_in_list, \
    lst_str_contexts_names, \
    DIR_PATH_MODELS_INPUT_CONFIGS, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS, \
    DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS


class ChatState :

    def __init__(self,) :
        self._prevLexicalEntry = None
        self._prevSense = None
        self._prevSynset = None

        self._currLexicalEntry = None
        self._currSense = None
        self._currSynset = None

    @property
    def previous_lexical_entry(self,) :
        return self._prevLexicalEntry

    @previous_lexical_entry.setter
    def previous_lexical_entry(self, value) :
        self._prevLexicalEntry = value

    @property
    def previous_sense(self,) :
        return self._prevSense

    @previous_sense.setter
    def previous_sense(self, value) :
        self._prevSense = value

    @property
    def previous_synset(self,) :
        return self._prevSynset

    @previous_synset.setter
    def previous_synset(self, value) :
        self._prevSynset = value

    @property
    def current_lexical_entry(self,) :
        return self._currLexicalEntry

    @current_lexical_entry.setter
    def current_lexical_entry(self, value) :
        self._currLexicalEntry = value

    @property
    def current_sense(self,) :
        return self._currSense

    @current_sense.setter
    def current_sense(self, value) :
        self._currSense = value

    @property
    def current_synset(self,) :
        return self._currSynset

    @current_synset.setter
    def current_synset(self, value) :
        self._currSynset = value


class Context :

    def __init__(self, str_context_name,) :
        self._str_context_name = str_context_name
        self._model = None
        self._model_string_inverse_lookup = None
        self._config = None

    def load(self,) :
        with open(os.path.join(
                DIR_PATH_MODELS_OUTPUT_INTENTS_SCHEMAS,
                self._str_context_name + '_model_intents_schema.json')
                ) as file_handle :
            self._model = model_from_json(json.load(file_handle))
        self._model.load_weights(os.path.join(
            DIR_PATH_MODELS_OUTPUT_INTENTS_PARAMS,
            self._str_context_name + '_model_intents_params.h5'))
        with open(os.path.join(
                DIR_PATH_MODELS_OUTPUT_LOOKUP_SCHEMAS,
                self._str_context_name + '_model_lookup_schema.json')
                ) as file_handle :
            self._model_string_inverse_lookup = model_from_json(json.load(
                file_handle))
        self._model_string_inverse_lookup.load_weights(os.path.join(
            DIR_PATH_MODELS_OUTPUT_LOOKUP_PARAMS,
            self._str_context_name + '_model_lookup_params.h5'))
        self._config = configparser.ConfigParser()
        self._config.read(os.path.join(
            DIR_PATH_MODELS_INPUT_CONFIGS, self._str_context_name + '.ini'))

    def ask_for_intent(
            self,
            str_intent_prompt = None,
            str_intent_feature = None,
            lst_str_expected_intents = None) :

        str_intent = None

        if str_intent_prompt is not None and str_intent_prompt != "" :
            print(Fore.MAGENTA + str_intent_prompt + Style.RESET_ALL)
        if str_intent_feature is None :
            str_orig_intent_feature = input().strip()
        else :
            str_orig_intent_feature = str_intent_feature

        if str_orig_intent_feature != "" :
            conf_prep = self._config["intent.features.preprocessing"]
            (lst_str_clean_intent_features, set_all_features_dictionary) = \
                clean_intent_features_in_list(
                    lst_str_intent_features = [str_orig_intent_feature],
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

            str_clean_intent_feature = lst_str_clean_intent_features[0]
            np_arr_flt_predicted_intent_probs = self._model.predict(
                tf.constant([str_clean_intent_feature]))
            np_arr_int_sorted_predicted_intent_indices = \
                tf.argsort(-np_arr_flt_predicted_intent_probs,
                           axis = -1).numpy()[0,:]
            np_arr_str_sorted_predicted_intent_labels = \
                self._model_string_inverse_lookup.predict(
                    np_arr_int_sorted_predicted_intent_indices)
            lst_str_sorted_predicted_intent_labels = [
                bin_label.decode('ascii') for bin_label in
                np_arr_str_sorted_predicted_intent_labels]
            lst_flt_predicted_intent_probs = list(
                np_arr_flt_predicted_intent_probs[0,:])
            lst_flt_sorted_predicted_intent_probs = [lst_flt_predicted_intent_probs[
                int_index] for int_index in np_arr_int_sorted_predicted_intent_indices]

            if False :
                print("Original Feature: " + str_orig_intent_feature)
                print("Clean Feature: " + str_clean_intent_feature)
                print("Level 1 Predicted Label: " +
                      lst_str_sorted_predicted_intent_labels[0] +
                      " (" + str(lst_flt_sorted_predicted_intent_probs[0]) + ")")
                print("Level 2 Predicted Label: " +
                      lst_str_sorted_predicted_intent_labels[1] +
                      " (" + str(lst_flt_sorted_predicted_intent_probs[1]) + ")")
                print("Level 3 Predicted Label: " +
                      lst_str_sorted_predicted_intent_labels[2] +
                      " (" + str(lst_flt_sorted_predicted_intent_probs[2]) + ")")
                print()

            if lst_str_expected_intents is None :
                lst_str_sorted_filtered_predicted_intent_labels = \
                    lst_str_sorted_predicted_intent_labels
                lst_flt_sorted_filtered_predicted_intent_probs = \
                    lst_flt_sorted_predicted_intent_probs
            else :
                lst_str_sorted_filtered_predicted_intent_labels = []
                lst_flt_sorted_filtered_predicted_intent_probs = []
                for (str_predicted_intent_label, flt_predicted_intent_prob) in zip(
                        lst_str_sorted_predicted_intent_labels,
                        lst_flt_sorted_predicted_intent_probs) :
                    if str_predicted_intent_label in lst_str_expected_intents :
                        lst_str_sorted_filtered_predicted_intent_labels.append(
                            str_predicted_intent_label)
                        lst_flt_sorted_filtered_predicted_intent_probs.append(
                            flt_predicted_intent_prob)

            if len(lst_str_sorted_filtered_predicted_intent_labels) == 0 :
                str_intent = None
            elif len(lst_str_sorted_filtered_predicted_intent_labels) == 1 :
                str_intent = lst_str_sorted_filtered_predicted_intent_labels[0]
            else :
                str_intent = random.choices(
                    population = lst_str_sorted_filtered_predicted_intent_labels,
                    weights = lst_flt_sorted_filtered_predicted_intent_probs,
                    k = 1,)[0]

        return str_intent


def initialize_chat() :

    # int_random_seed = 12345
    int_random_seed = datetime.now().timestamp()
    random.seed(int_random_seed)

    chat_state = ChatState()

    dict_all_context_names_to_contexts = {}

    colorama.init()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print(
        Fore.CYAN +
        "Let me introduce myself: I am the WordNet ChatBot." +
        Style.RESET_ALL)
    print(Fore.CYAN +
          "During our upcoming chat, we will dive into a huge and complex world of words." +
          Style.RESET_ALL)
    print(
        Fore.CYAN +
        "Please, give me a few seconds to refresh my memory before we start chatting ..." +
        Style.RESET_ALL)
    lexicon = Lexicon.load_from_lzma(
        str_file_path = DIR_PATH_DATA_PKL_XZ_WORDNET,
        str_file_name = FILE_NAME_DATA_PKL_XZ_WORDNET)
    # lexicon.print_summary()
    for str_context_name in lst_str_contexts_names :
        dict_all_context_names_to_contexts[str_context_name] = Context(
            str_context_name = str_context_name)
        dict_all_context_names_to_contexts[str_context_name].load()
    print(Fore.CYAN + "I am ready to chat with you now." + Style.RESET_ALL)

    return (lexicon, chat_state, dict_all_context_names_to_contexts)


def ask_to_finish_chat(dict_all_context_names_to_contexts) :

    bool_chat_end_intent = False

    if False :
        print(Fore.MAGENTA + "Do you still want to continue our chat?" + Style.RESET_ALL)
        str_yes_no = input().strip().lower()
    else :
        str_yes_no = dict_all_context_names_to_contexts[
            "context_08_yes_no"].ask_for_intent(
                str_intent_prompt = "Do you still want to continue our chat?",
                str_intent_feature = None,
                lst_str_expected_intents = None,)
    if str_yes_no == "yes" :
        print(Fore.GREEN + "Great! Let's continue chatting." + Style.RESET_ALL)
    elif str_yes_no == "no" :
        if random.randint(1, 2) == 1 :
            print(Fore.GREEN + "I truly enjoyed chatting with you. Goodbye." + Style.RESET_ALL)
        else :
            print(Fore.GREEN + "It was nice chatting with you. Farewell." + Style.RESET_ALL)
        bool_chat_end_intent = True
    else :
        print(Fore.RED + 'I did not get it.' + Style.RESET_ALL)
        print(Fore.CYAN + 'I just expected simple "yes" or "no" from you.' + Style.RESET_ALL)
        print(Fore.CYAN + 'Please, be direct and concise in your responses next time.' + Style.RESET_ALL)
        print(Fore.CYAN + 'You may always press Ctrl + C to finish our chat.' + Style.RESET_ALL)

    return bool_chat_end_intent


def ask_whether_to_replace_lexical_entry(
        chat_state, dict_all_context_names_to_contexts) :

    bool_replace_lexical_entry = False

    if False :
        print(
            Fore.MAGENTA +
            'Would like to replace {0} "{1}" with another lexical entry?'.format(
                dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
        str_yes_no = input().strip().lower()
    else :
        str_yes_no = dict_all_context_names_to_contexts[
            "context_08_yes_no"].ask_for_intent(
                str_intent_prompt = 'Would like to replace {0} "{1}" with another lexical entry?'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form),
                str_intent_feature = None,
                lst_str_expected_intents = None,)

    if str_yes_no == "yes" :
        bool_replace_lexical_entry = True
        print(Fore.GREEN + "Fine! Let's replace it ..." + Style.RESET_ALL)
    elif str_yes_no == "no" :
        print(Fore.GREEN + "No problem. Let's keep the current one." + Style.RESET_ALL)
    else :
        print(Fore.RED + 'I did not get it, but I will take your response as "no".' + Style.RESET_ALL)
        print(Fore.CYAN + 'I just expected simple "yes" or "no" from you.' + Style.RESET_ALL)
        print(Fore.CYAN + 'Please, be direct and concise in your responses next time.' + Style.RESET_ALL)

    return bool_replace_lexical_entry


def find_lexical_entry(
        lexicon, chat_state, dict_all_context_names_to_contexts) :

    found_LexicalEntry = None

    if False :
        print(
            Fore.MAGENTA +
            'Do you have any specific word on your mind?' +
            Style.RESET_ALL)
        str_yes_no = input().strip().lower()
    else :
        str_yes_no = dict_all_context_names_to_contexts[
            "context_08_yes_no"].ask_for_intent(
                str_intent_prompt = 'Do you have any SPECIFIC word on your mind?',
                str_intent_feature = None,
                lst_str_expected_intents = None,)

    if str_yes_no == "yes" :

        print(Fore.MAGENTA + "What is your choice of a written word?" + Style.RESET_ALL)
        str_writtenForm = input().strip()
        lst_found_LexicalEntry = []
        for le in lexicon.dictionary_of_lexical_entries.values() :
            if str_writtenForm == le.written_form :
                lst_found_LexicalEntry.append(le)

        if len(lst_found_LexicalEntry) == 0 :
            # Try using fuzzy matching
            int_top_best_matches_max_count = 7
            if int_top_best_matches_max_count == 1 :
                # This code is inactive (see "int_top_best_matches_max_count")
                str_best_matching_written_form = None
                flt_dam_lev_norm_dist_upper_threshold = .5 # up to 1.
                flt_shortest_dam_lev_norm_dist = flt_dam_lev_norm_dist_upper_threshold
                for le in lexicon.dictionary_of_lexical_entries.values() :
                    flt_curr_dam_lev_norm_dist = DamerauLevenshtein.normalized_distance(
                        str_writtenForm, le.written_form)
                    if flt_curr_dam_lev_norm_dist < flt_shortest_dam_lev_norm_dist :
                        flt_shortest_dam_lev_norm_dist = flt_curr_dam_lev_norm_dist
                        str_best_matching_written_form = le.written_form
                if str_best_matching_written_form is not None :
                    print(
                        Fore.RED +
                        'I have never heard of "{0}", but it looks similar to "{1}".'.format(
                            str_writtenForm,
                            str_best_matching_written_form,) +
                        Style.RESET_ALL)

                    if False :
                        print(Fore.MAGENTA + 'Did you actually mean "{0}"?'.format(
                            str_best_matching_written_form) + Style.RESET_ALL)
                        str_yes_no = input().strip().lower()
                    else :
                        str_yes_no = dict_all_context_names_to_contexts[
                            "context_08_yes_no"].ask_for_intent(
                                str_intent_prompt = 'Did you actually mean "{0}"?'.format(
                                    str_best_matching_written_form),
                                str_intent_feature = None,
                                lst_str_expected_intents = None,)

                    if str_yes_no == "yes" :
                        str_writtenForm = str_best_matching_written_form
                        for le in lexicon.dictionary_of_lexical_entries.values() :
                            if str_writtenForm == le.written_form :
                                lst_found_LexicalEntry.append(le)
                    elif str_yes_no == "no" :
                        pass
                    else :
                        print(Fore.RED + 'You response is unclear to me, but I will take it for "no".' +
                              Style.RESET_ALL)
            else : # if int_top_best_matches_max_count > 1
                str_best_matching_written_form = None
                lst_tpl_best_matching_written_forms = []
                for le in lexicon.dictionary_of_lexical_entries.values() :
                    flt_curr_dam_lev_norm_dist = DamerauLevenshtein.normalized_distance(
                        str_writtenForm, le.written_form)
                    if len(lst_tpl_best_matching_written_forms) < int_top_best_matches_max_count :
                        heapq.heappush(
                            lst_tpl_best_matching_written_forms,
                            (-flt_curr_dam_lev_norm_dist, le.written_form))
                    elif lst_tpl_best_matching_written_forms[0][0] < flt_curr_dam_lev_norm_dist :
                        heapq.heappushpop(
                            lst_tpl_best_matching_written_forms,
                            (-flt_curr_dam_lev_norm_dist, le.written_form))
                intNumWrittenForms = len(lst_tpl_best_matching_written_forms)
                lst_str_best_matching_written_forms = [None] * intNumWrittenForms
                for i in range(len(lst_str_best_matching_written_forms)-1,-1,-1) :
                    lst_str_best_matching_written_forms[i] = \
                        heapq.heappop(lst_tpl_best_matching_written_forms)[1]
                print(
                    Fore.RED +
                    'I have never heard of "{0}", but it looks similar to one of the following options:'.format(
                        str_writtenForm,) + Style.RESET_ALL)
                for i in range(len(lst_str_best_matching_written_forms)) :
                    print(Fore.CYAN + '{0}. "{1}"'.format(
                        i + 1, lst_str_best_matching_written_forms[i],) + Style.RESET_ALL)
                print(
                    Fore.MAGENTA +
                    "Which word did you have in mind, from 1 to {0}?".format(intNumWrittenForms) +
                    "\nYou may stay silent, if you meant none of them." +
                    Style.RESET_ALL)
                strSelectedWrittenFormIndex = input().strip()
                if strSelectedWrittenFormIndex != "" :
                    strSelectedWrittenFormIndex = \
                        dict_all_context_names_to_contexts[
                            "context_09_positive_integers"].ask_for_intent(
                                str_intent_prompt = None,
                                str_intent_feature = strSelectedWrittenFormIndex,
                                lst_str_expected_intents = [str(i+1) for i in range(intNumWrittenForms)])
                if strSelectedWrittenFormIndex == "" :
                    str_best_matching_written_form = None
                elif strSelectedWrittenFormIndex.isdigit() :
                    intSelectedWrittenFormIndex = int(strSelectedWrittenFormIndex) - 1
                    if 0 <= intSelectedWrittenFormIndex < intNumWrittenForms :
                        str_best_matching_written_form = lst_str_best_matching_written_forms[
                            intSelectedWrittenFormIndex]
                        str_writtenForm = str_best_matching_written_form
                        for le in lexicon.dictionary_of_lexical_entries.values() :
                            if str_writtenForm == le.written_form :
                                lst_found_LexicalEntry.append(le)
                    else :
                        print(
                            Fore.RED +
                            "I expected the sense index to be from 1 to {0}.".format(
                                intNumWrittenForms)  + Style.RESET_ALL)
                else :
                    print(Fore.RED +
                           "I do not get it. I expected either an empty response or a valid integer." +
                           Style.RESET_ALL)

        if len(lst_found_LexicalEntry) == 0 :
            print(Fore.RED + 'I have no idea what "{0}" is. Please, ask again.'.format(
                str_writtenForm) + Style.RESET_ALL)
        else :
            if len(lst_found_LexicalEntry) == 1 :
                found_LexicalEntry = lst_found_LexicalEntry[0]
                str_selected_part_of_speech_code = found_LexicalEntry.part_of_speech
                print(Fore.GREEN + 'I know the {0} "{1}".'.format(
                    dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                    found_LexicalEntry.written_form,) + Style.RESET_ALL)
                chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                chat_state.current_lexical_entry = found_LexicalEntry
                chat_state.previous_sense = chat_state.current_sense
                chat_state.current_sense = None
                chat_state.previous_synset = chat_state.current_synset
                chat_state.current_synset = None
            else :
                str_parts_of_speech = ', '.join([
                    '"' + dict_parts_of_speech_code_to_name[
                        iter_LexicalEntry.part_of_speech] + '"'
                    for iter_LexicalEntry in lst_found_LexicalEntry])
                print(
                    Fore.GREEN +
                    'This word "{0}" is used as these {1} parts of speech: {2}.'.format(
                    str_writtenForm,
                    len(lst_found_LexicalEntry),
                    str_parts_of_speech) + Style.RESET_ALL)
                lst_str_candidate_parts_of_speech_codes = [
                    iter_LexicalEntry.part_of_speech for iter_LexicalEntry in
                    lst_found_LexicalEntry]
                str_selected_part_of_speech_code = \
                    dict_all_context_names_to_contexts[
                        "context_01_parts_of_speech"].ask_for_intent(
                    str_intent_prompt = "Which part of speech are you interested in?",
                    str_intent_feature = None,
                    lst_str_expected_intents = lst_str_candidate_parts_of_speech_codes)
                for i in range(len(lst_found_LexicalEntry)) :
                    if lst_found_LexicalEntry[i].part_of_speech == \
                       str_selected_part_of_speech_code :
                        found_LexicalEntry = lst_found_LexicalEntry[i]
                        break
                if found_LexicalEntry is not None :
                    chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                    chat_state.current_lexical_entry = found_LexicalEntry
                    chat_state.previous_sense = chat_state.current_sense
                    chat_state.current_sense = None
                    chat_state.previous_synset = chat_state.current_synset
                    chat_state.current_synset = None
                    print(
                        Fore.GREEN +
                        'As I understand, your choice is the {0} "{1}".'.format(
                        dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                        found_LexicalEntry.written_form) + Style.RESET_ALL)
                else :
                    print(
                        Fore.RED +
                        'I do not know any {0} "{1}". Please, try again.'.format(
                        str_parts_of_speech, str_writtenForm) + Style.RESET_ALL)

    elif str_yes_no == "no" :

        print(Fore.CYAN + "In this case, I will make a choice for you."  + Style.RESET_ALL)
        str_context_name = "context_01_parts_of_speech"

        str_selected_part_of_speech_code = \
            dict_all_context_names_to_contexts[
                str_context_name].ask_for_intent(
            str_intent_prompt = "Which part of speech are you interested in: {0}?".format(
                ", ".join(list(dict_parts_of_speech_code_to_name.values()))),
            str_intent_feature = None,
            lst_str_expected_intents = None,)
        if str_selected_part_of_speech_code is None :
            lst_parts_of_speech_codes = list(dict_parts_of_speech_code_to_name.keys())
            str_selected_part_of_speech_code = lst_parts_of_speech_codes[
                random.randint(0, len(lst_parts_of_speech_codes) - 1)]
            print(Fore.RED + 'You choice is unclear to me, and I choose part of speech "{0}" for you.'.format(
                dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                ) + Style.RESET_ALL)

        lst_found_LexicalEntry = []
        for le in lexicon.dictionary_of_lexical_entries.values() :
            if str_selected_part_of_speech_code == le.part_of_speech :
                lst_found_LexicalEntry.append(le)
        if len(lst_found_LexicalEntry) == 0 :
            print(Fore.RED + 'Sorry, but I do not know any such part of speech.' +
                Style.RESET_ALL)
        else : # len(lst_found_LexicalEntry) >= 1 :
            print(Fore.GREEN + 'I am aware of {0} {1}{2}.'.format(
                len(lst_found_LexicalEntry),
                dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                "" if len(lst_found_LexicalEntry) == 1 else "s") +
                Style.RESET_ALL)
            if str_selected_part_of_speech_code in {'n', 'a', 'v'} :
                if False :
                    print(Fore.MAGENTA + "Do you have any {0}-specific word category on your mind?".format(
                        dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],) +
                        Style.RESET_ALL)
                    str_yes_no = input().strip().lower()
                else :
                    str_yes_no = dict_all_context_names_to_contexts[
                        "context_08_yes_no"].ask_for_intent(
                            str_intent_prompt = "Do you have any {0}-specific word category on your mind?".format(
                                dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],),
                            str_intent_feature = None,
                            lst_str_expected_intents = None,)

                if str_yes_no == "yes" :
                    str_context_name = None
                    if str_selected_part_of_speech_code == 'n' :
                        str_context_name = "context_04_synset_lexicographer_noun_file"
                        dict_synset_lexicographer_files_name_to_descr = \
                            dict_synset_lexicographer_noun_files_name_to_descr
                    elif str_selected_part_of_speech_code == 'a' :
                        str_context_name = "context_05_synset_lexicographer_adj_file"
                        dict_synset_lexicographer_files_name_to_descr = \
                            dict_synset_lexicographer_adj_files_name_to_descr
                    elif str_selected_part_of_speech_code == 'v' :
                        str_context_name = "context_06_synset_lexicographer_verb_file"
                        dict_synset_lexicographer_files_name_to_descr = \
                            dict_synset_lexicographer_verb_files_name_to_descr
                    lst_str_shuffled_lex_files_keys = \
                        list(dict_synset_lexicographer_files_name_to_descr.keys())
                    random.shuffle(lst_str_shuffled_lex_files_keys)

                    int_lex_files_count = len(lst_str_shuffled_lex_files_keys)
                    int_num_lex_file_examples = 3
                    int_start_idx = 0
                    int_inc_end_idx = min(int_lex_files_count, int_start_idx + int_num_lex_file_examples)
                    bool_finished_batches_of_hints = False
                    str_more = ""
                    while not bool_finished_batches_of_hints :
                        print(Fore.CYAN + "These are a few {0}examples of lexicographer files for {1}s:".format(
                            str_more,
                            dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                            ) + Style.RESET_ALL)
                        str_more = "more "
                        for i in range(int_start_idx, int_inc_end_idx) :
                            str_current_lexicographer_file_code = lst_str_shuffled_lex_files_keys[i]
                            print(Fore.CYAN + '{0}. "{1}": {2}.'.format(
                                i - int_start_idx + 1,
                                str_current_lexicographer_file_code.split(".")[-1],
                                dict_synset_lexicographer_files_name_to_descr[str_current_lexicographer_file_code],
                                ) + Style.RESET_ALL)
                        int_start_idx = int_start_idx + int_num_lex_file_examples
                        int_inc_end_idx = min(int_lex_files_count, int_start_idx + int_num_lex_file_examples)
                        if int_start_idx >= int_lex_files_count :
                            bool_finished_batches_of_hints = True
                        print(Fore.MAGENTA + "What is your choice of lexicographer file for {0}s?{1}".format(
                            dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                            " I've already provided all hints to you."
                            if bool_finished_batches_of_hints else
                            " You may stay silent to get more options as hints.",
                            ) + Style.RESET_ALL)
                        str_response_on_lexicographer_file = input().strip()
                        if str_response_on_lexicographer_file != "" :
                            bool_finished_batches_of_hints = True
                    str_selected_lexicographer_file_code = dict_all_context_names_to_contexts[
                        str_context_name].ask_for_intent(
                            str_intent_prompt = "",
                            str_intent_feature = str_response_on_lexicographer_file,
                            lst_str_expected_intents = None,)
                    print(Fore.GREEN + 'I will consider your choice of "{0}" lexicographer file, which includes {1}.'.format(
                        str_selected_lexicographer_file_code.split(".")[-1],
                        dict_synset_lexicographer_files_name_to_descr[str_selected_lexicographer_file_code],) + Style.RESET_ALL)

                    lst_orig_found_LexicalEntry = lst_found_LexicalEntry
                    lst_found_LexicalEntry = []
                    for tmp_LexicalEntry in lst_orig_found_LexicalEntry :
                        bool_lex_entry_has_lex_file_categ = False
                        for tmp_Sense in list(tmp_LexicalEntry.dictionary_of_senses.values()) :
                            if lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier].lexical_file_category == str_selected_lexicographer_file_code :
                                bool_lex_entry_has_lex_file_categ = True
                                break
                        if bool_lex_entry_has_lex_file_categ : 
                            lst_found_LexicalEntry.append(tmp_LexicalEntry)
                    print(Fore.CYAN + 'I know {0} {1}{2} from the lexicographer file "{3}", which has {4}!'.format(
                        len(lst_found_LexicalEntry),
                        dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                        "" if len(lst_found_LexicalEntry) == 1 else "s",
                        str_selected_lexicographer_file_code.split(".")[-1],
                        dict_synset_lexicographer_files_name_to_descr[str_selected_lexicographer_file_code],
                        ) + Style.RESET_ALL)
                elif str_yes_no == "no" :
                    print(Fore.CYAN + "In this case, I will make a choice from any word category." + Style.RESET_ALL)
                else :
                    print(Fore.RED + 'I did not get it. I expected "yes" or "no" from you.' + Style.RESET_ALL)
                    print(Fore.RED + 'But I will take your answer as "no", and I will make a choice from any word category.' + Style.RESET_ALL)

            intSelectedLexicalEntryIndex = random.randint(0, len(lst_found_LexicalEntry) - 1)
            found_LexicalEntry = lst_found_LexicalEntry[intSelectedLexicalEntryIndex]
            chat_state.previous_lexical_entry = chat_state.current_lexical_entry
            chat_state.current_lexical_entry = found_LexicalEntry
            chat_state.previous_sense = chat_state.current_sense
            chat_state.current_sense = None
            chat_state.previous_synset = chat_state.current_synset
            chat_state.current_synset = None
            print(
                Fore.GREEN +
                'My choice for you is the {0} "{1}".'.format(
                dict_parts_of_speech_code_to_name[str_selected_part_of_speech_code],
                found_LexicalEntry.written_form) + Style.RESET_ALL)
    else :
        print(Fore.RED + 'I did not get it. I expected "yes" or "no" from you.' + Style.RESET_ALL)


def process_found_lexical_entry(
        found_LexicalEntry, lexicon, chat_state, dict_all_context_names_to_contexts) :

    updated_LexicalEntry = None

    if chat_state.current_lexical_entry != found_LexicalEntry :
        chat_state.previous_lexical_entry = chat_state.current_lexical_entry
        chat_state.current_lexical_entry = found_LexicalEntry

    intNumPronunciations = len(chat_state.current_lexical_entry.lemma.list_of_pronunciations)
    if intNumPronunciations == 1 :
        str_pronunciation = chat_state.current_lexical_entry.lemma.list_of_pronunciations[0].text
        str_variety = chat_state.current_lexical_entry.lemma.list_of_pronunciations[0].variety
        print(
            Fore.CYAN +
            'The {0} "{1}" is pronounced as follows: [{2}]{3}.'.format(
            dict_parts_of_speech_code_to_name[
                chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            str_pronunciation,
            "" if str_variety is None else ' ("{0}")'.format(str_variety),
            ) + Style.RESET_ALL)
    elif intNumPronunciations > 1 :
        str_pronunciations = '"' + '", or "'.join(
            ["[" + pronunciation_obj.text + "]" +
             ("" if pronunciation_obj.variety is None else ' ("{0}")'.format(
                 pronunciation_obj.variety))
             for pronunciation_obj in chat_state.current_lexical_entry.lemma.list_of_pronunciations]) + '"'
        print(
            Fore.CYAN +
            'The {0} "{1}" is pronounced in {2} ways: {3}.'.format(
            dict_parts_of_speech_code_to_name[
                chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            intNumPronunciations, str_pronunciations) + Style.RESET_ALL)

    intNumForms = len(chat_state.current_lexical_entry.list_of_forms)
    if intNumForms == 1 :
        print(
            Fore.CYAN +
            'The {0} "{1}" has an alternative form, such as "{2}".'.format(
            dict_parts_of_speech_code_to_name[
                chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            chat_state.current_lexical_entry.list_of_forms[0].written_form,) + Style.RESET_ALL)
    elif intNumForms > 1 :
        str_forms = '"' + '", or "'.join(
            [form_obj.written_form for form_obj in chat_state.current_lexical_entry.list_of_forms]) + '"'
        print(
            Fore.CYAN +
            'The {0} "{1}" has {2} more alternative forms: {3}.'.format(
            dict_parts_of_speech_code_to_name[
                chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            intNumForms, str_forms) + Style.RESET_ALL)

    intNumSenses = len(chat_state.current_lexical_entry.dictionary_of_senses)
    if intNumSenses >= 1 :
        if intNumSenses == 1 :
            tmp_Sense = list(chat_state.current_lexical_entry.dictionary_of_senses.values())[0]
            tmp_Synset = lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier]
            str_definitions = '"' + '", or "'.join(
                [definition.text for definition in tmp_Synset.list_of_definitions]) + '"'
            if str_definitions == '""' :
                str_definitions = "no definition"
            print(
                Fore.CYAN +
                'The {0} "{1}" has just one sense, such as {2}.'.format(
                dict_parts_of_speech_code_to_name[
                    chat_state.current_lexical_entry.part_of_speech],
                chat_state.current_lexical_entry.written_form,
                str_definitions,) + Style.RESET_ALL)
            intSelectedSenseIndex = 0
        elif intNumSenses > 1 :
            print(
                Fore.CYAN +
                'The {0} "{1}" has {2} senses:'.format(
                dict_parts_of_speech_code_to_name[
                    chat_state.current_lexical_entry.part_of_speech],
                chat_state.current_lexical_entry.written_form,
                intNumSenses) + Style.RESET_ALL)
            lst_senses_for_curr_lexical_entry = list(
                chat_state.current_lexical_entry.dictionary_of_senses.values())
            for (i, tmp_Sense) in enumerate(lst_senses_for_curr_lexical_entry) :
                tmp_Synset = lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier]
                str_definitions = '"' + '", or "'.join(
                    [definition_obj.text for definition_obj in tmp_Synset.list_of_definitions]) + '"'
                if str_definitions == '""' :
                    str_definitions = "no definition"
                print(Fore.CYAN +'{0}. {1}.'.format(
                    i + 1, str_definitions) + Style.RESET_ALL)

            boolIsSelectedSenseIndexValid = False
            while not boolIsSelectedSenseIndexValid :
                print(
                    Fore.MAGENTA +
                    "Which sense are you interested in, from 1 to {0}?".format(intNumSenses) +
                    "\nYou can stay silent if you are interested in any random sense." +
                    Style.RESET_ALL)
                strSelectedSenseIndex = input().strip()
                if strSelectedSenseIndex != "" :
                    strSelectedSenseIndex = \
                        dict_all_context_names_to_contexts[
                            "context_09_positive_integers"].ask_for_intent(
                                str_intent_prompt = None,
                                str_intent_feature = strSelectedSenseIndex,
                                lst_str_expected_intents = [str(i+1) for i in range(intNumSenses)])
                if strSelectedSenseIndex == "" :
                    intSelectedSenseIndex = random.randint(0, intNumSenses - 1)
                    boolIsSelectedSenseIndexValid = True
                elif strSelectedSenseIndex.isdigit() :
                    intSelectedSenseIndex = int(strSelectedSenseIndex) - 1
                    if 0 <= intSelectedSenseIndex < intNumSenses :
                        boolIsSelectedSenseIndexValid = True
                    else :
                        print(Fore.RED + "I expected the sense index to be from 1 to {0}.".format(intNumSenses)  + Style.RESET_ALL)
                else :
                    print(Fore.RED +
                           "I do not get it. I expected either an empty response or a valid integer." + Style.RESET_ALL)
        chat_state.previous_sense = chat_state.current_sense
        chat_state.current_sense = list(chat_state.current_lexical_entry.dictionary_of_senses.values())[
            intSelectedSenseIndex]
        chat_state.previous_synset = chat_state.current_synset
        chat_state.current_synset = lexicon.dictionary_of_synsets[chat_state.current_sense.synset_identifier]

        if intNumSenses > 1 :
            print(
                Fore.GREEN +
                "Let me tell you about sense # {0}.".format(
                intSelectedSenseIndex + 1,) + Style.RESET_ALL)

        intNumSenseRelations = len(chat_state.current_sense.dictionary_of_sense_relations)
        if intNumSenseRelations == 0 :
            print(
                Fore.CYAN +
                "This sense has no relations with other senses." + Style.RESET_ALL)
        elif intNumSenseRelations == 1 :
            print(
                Fore.CYAN +
                "This sense has a relation with another sense." + Style.RESET_ALL)
        elif intNumSenseRelations > 1 :
            print(
                Fore.CYAN +
                "This sense has {0} relations with other senses.".format(
                intNumSenseRelations,) + Style.RESET_ALL)

        int_syntactic_behaviour_count = len(chat_state.current_sense.list_of_syntactic_behaviour_identifiers)
        if int_syntactic_behaviour_count == 1 :
            str_sb_id = chat_state.current_sense.list_of_syntactic_behaviour_identifiers[0]
            print(Fore.CYAN +
                  'This sense of the {0} "{1}" has "{2}" syntactic behaviour: [{3}], e.g. "{4}".'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form,
                   str_sb_id,
                   dict_sense_relations_syntactic_behaviours_code_to_descr[str_sb_id],
                   lexicon.dictionary_of_syntactic_behaviours[
                       str_sb_id].subcategorization_frame.strip("()"),) + Style.RESET_ALL)
        elif int_syntactic_behaviour_count > 1 :
            print(Fore.CYAN +
                  'This sense of the {0} "{1}" has these syntactic behaviours:'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)
            for (i, str_sb_id) in enumerate(chat_state.current_sense.list_of_syntactic_behaviour_identifiers) :
                print(Fore.CYAN + '{0}. "{1}": [{2}], e.g. "{3}".'.format(
                    i + 1, str_sb_id,
                    dict_sense_relations_syntactic_behaviours_code_to_descr[str_sb_id],
                    lexicon.dictionary_of_syntactic_behaviours[
                        str_sb_id].subcategorization_frame.strip("()")) + Style.RESET_ALL)

        int_lexical_entries_count = len(chat_state.current_synset.list_of_lexical_entries_identifiers)
        if int_lexical_entries_count == 0 or (
                int_lexical_entries_count == 1 and
                chat_state.current_lexical_entry.identifier !=
                chat_state.current_synset.list_of_lexical_entries_identifiers[0]) :
            print(Fore.RED +
                  'Sorry, but I am confused about this sense of the {0} "{1}" and its synonyms.'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
        elif int_lexical_entries_count == 1 :
            print(Fore.CYAN +
                  'This sense of the {0} "{1}" has no synonyms.'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
        else : # int_lexical_entries_count > 1
            print(Fore.CYAN +
                  'This sense of the {0} "{1}" is a part of the following set of synonyms:'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
        if int_lexical_entries_count > 1 :
            for (int_SynonymIndex, str_lexical_entry_identifier) in \
               enumerate(chat_state.current_synset.list_of_lexical_entries_identifiers) :
                tmp_LexicalEntry = lexicon.dictionary_of_lexical_entries[
                    str_lexical_entry_identifier]
                print(Fore.CYAN + '{0}. {1} ({2})'.format(
                    int_SynonymIndex + 1,
                    tmp_LexicalEntry.written_form,
                    dict_parts_of_speech_code_to_name[
                        tmp_LexicalEntry.part_of_speech]) + Style.RESET_ALL)

        intNumSynsetRelations = len(chat_state.current_synset.dictionary_of_synset_relations)
        if intNumSynsetRelations == 0 :
            print(
                Fore.CYAN +
                "This synonym group has no relations with other synonym groups." +
                Style.RESET_ALL)
        elif intNumSynsetRelations == 1 :
            print(
                Fore.CYAN +
                "This synonym group has a relation with another synonym group." +
                Style.RESET_ALL)
        elif intNumSynsetRelations > 1 :
            print(
                Fore.CYAN +
                "This synonym group has {0} relations with other synonym groups.".format(
                intNumSynsetRelations,) + Style.RESET_ALL)

        lfc = chat_state.current_synset.lexical_file_category
        str_lexical_file_category_descr = \
            dict_synset_lexicographer_noun_files_name_to_descr.get(lfc)
        if str_lexical_file_category_descr is None :
            str_lexical_file_category_descr = \
                dict_synset_lexicographer_adj_files_name_to_descr.get(lfc)
        if str_lexical_file_category_descr is None :
            str_lexical_file_category_descr = \
                dict_synset_lexicographer_verb_files_name_to_descr.get(lfc)
        if str_lexical_file_category_descr is not None :
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + 'This sense of the {0} "{1}" is in the lexical group of {2}.'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    str_lexical_file_category_descr) + Style.RESET_ALL)
            elif int_lexical_entries_count > 1 :
                print(Fore.CYAN + "The above synonyms are the lexical group of {0}.".format(
                    str_lexical_file_category_descr) + Style.RESET_ALL)

        if len(chat_state.current_synset.list_of_definitions) == 0 :
            if int_lexical_entries_count <= 1 :
                print(Fore.RED + "Unfortunately, I have no idea how to define this word." + Style.RESET_ALL)
            else :
                print(Fore.RED + "Unfortunately, I have no idea how to define this set of synonyms." + Style.RESET_ALL)
        elif len(chat_state.current_synset.list_of_definitions) == 1 :
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + 'This sense of the {0} "{1}" is defined as follows: "{2}".'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    chat_state.current_synset.list_of_definitions[0].text,) + Style.RESET_ALL)
            else :
                print(Fore.CYAN + 'The synonym group with the {0} "{1}" is defined as follows: "{2}".'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    chat_state.current_synset.list_of_definitions[0].text,) + Style.RESET_ALL)
        else : # len(chat_state.current_synset.list_of_definitions) > 1
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + 'This sense of the {0} "{1}" can be defined in different ways as follows:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)
            else :
                print(Fore.CYAN + 'The synonym group with the {0} "{1}" can be defined in different ways as follows:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)
            for (int_def_idx, definition_obj) in enumerate(chat_state.current_synset.list_of_definitions) :
                print(Fore.CYAN + '{0}. "{1}".'.format(
                    int_def_idx + 1, definition_obj.text) +
                    Style.RESET_ALL)

        if len(chat_state.current_synset.list_of_examples) == 0 :
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + "I wish I had examples for this word, but I don't." + Style.RESET_ALL)
            else :
                print(Fore.CYAN + "I wish I had examples for this set of synonyms, but I don't." + Style.RESET_ALL)
        elif len(chat_state.current_synset.list_of_examples) == 1 :
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + 'This sense of the {0} "{1}" has an example as follows: "{2}".'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    chat_state.current_synset.list_of_examples[0].text,) + Style.RESET_ALL)
            else :
                print(Fore.CYAN + 'The synonym group with the {0} "{1}" has an example as follows: "{2}".'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    chat_state.current_synset.list_of_examples[0].text,) + Style.RESET_ALL)
        else : # len(chat_state.current_synset.list_of_examples) > 1
            if int_lexical_entries_count <= 1 :
                print(Fore.CYAN + 'This sense of the {0} "{1}" has some examples as follows:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)
            else :
                print(Fore.CYAN + 'The synonym group with the {0} "{1}" has some examples as follows:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)
            for (int_exa_idx, example_obj) in enumerate(chat_state.current_synset.list_of_examples) :
                print(Fore.CYAN + '{0}. "{1}".'.format(
                    int_exa_idx + 1, example_obj.text) +
                    Style.RESET_ALL)

        if intNumSenses > 1 :
            int_num_of_attempts_to_ask = 2
            boolIsValidResponseProvided = False
            while not boolIsValidResponseProvided and int_num_of_attempts_to_ask > 0 :
                if intNumSenses == 2 :
                    if False :
                        print(
                            Fore.MAGENTA +
                            'Would you like to explore the other sense of the {0} "{1}"?'.format(
                                dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                                chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
                        str_yes_no = input().strip().lower()
                    else :
                        str_yes_no = dict_all_context_names_to_contexts[
                            "context_08_yes_no"].ask_for_intent(
                                str_intent_prompt = 'Would you like to explore the other sense of the {0} "{1}"?'.format(
                                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                                    chat_state.current_lexical_entry.written_form),
                                str_intent_feature = None,
                                lst_str_expected_intents = None,)
                else :
                    if False :
                        print(
                            Fore.MAGENTA +
                            'Would you like to explore more of the {0} senses of the {1} "{2}"?'.format(
                                intNumSenses,
                                dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                                chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
                        str_yes_no = input().strip().lower()
                    else :
                        str_yes_no = dict_all_context_names_to_contexts[
                            "context_08_yes_no"].ask_for_intent(
                                str_intent_prompt = 'Would you like to explore more of {0} senses of the {1} "{2}"?'.format(
                                    intNumSenses,
                                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                                    chat_state.current_lexical_entry.written_form),
                                str_intent_feature = None,
                                lst_str_expected_intents = None,)

                if str_yes_no == "yes" :
                    chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                    updated_LexicalEntry = chat_state.current_lexical_entry
                    chat_state.previous_sense = chat_state.current_sense
                    chat_state.current_sense = None
                    chat_state.previous_synset = chat_state.current_synset
                    chat_state.current_synset = None
                    boolIsValidResponseProvided = True
                    print(Fore.GREEN +
                          "I think that you have a great idea to explore other senses." +
                          Style.RESET_ALL)
                elif str_yes_no == "no" :
                    updated_LexicalEntry = None
                    boolIsValidResponseProvided = True
                    print(Fore.GREEN +
                          "In this case, let's discuss something else ..." +
                          Style.RESET_ALL)
                elif int_num_of_attempts_to_ask > 1 :
                    int_num_of_attempts_to_ask = int_num_of_attempts_to_ask - 1
                    print(
                        Fore.RED +
                        'I do not understand your point. I expected simple "yes" or "no" from you.' +
                        Style.RESET_ALL)
                else :
                    updated_LexicalEntry = None
                    boolIsValidResponseProvided = True
                    print(Fore.RED + 'Fine! I will take it for "no".' + Style.RESET_ALL)
                    print(Fore.CYAN + "Let's discuss something else ..." + Style.RESET_ALL)
        else :
            updated_LexicalEntry = None

        if updated_LexicalEntry is None and int_lexical_entries_count > 1 :
            print(Fore.CYAN +
                  'As you may recall, this sense of the {0} "{1}" is a part of the following set of synonyms:'.format(
                   dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                   chat_state.current_lexical_entry.written_form) + Style.RESET_ALL)
            for (int_SynonymIndex, str_lexical_entry_identifier) in \
               enumerate(chat_state.current_synset.list_of_lexical_entries_identifiers) :
                tmp_LexicalEntry = lexicon.dictionary_of_lexical_entries[
                    str_lexical_entry_identifier]
                print(Fore.CYAN + '{0}. {1} ({2})'.format(
                    int_SynonymIndex + 1,
                    tmp_LexicalEntry.written_form,
                    dict_parts_of_speech_code_to_name[
                        tmp_LexicalEntry.part_of_speech]) + Style.RESET_ALL)
            boolIsSelectedLexicalEntryIndexValid = False
            while not boolIsSelectedLexicalEntryIndexValid :
                print(
                    Fore.MAGENTA +
                    "Would you be interested in discussing another word, from 1 to {0}?".format(int_lexical_entries_count) +
                    '\nYou can stay silent to let me choose for you, or say "no" in case of the lack of interest.' +
                    Style.RESET_ALL)
                strSelectedLexicalEntryIndex = input().strip().lower()
                if strSelectedLexicalEntryIndex != "" and strSelectedLexicalEntryIndex != "no" :
                    strSelectedLexicalEntryIndex = \
                        dict_all_context_names_to_contexts[
                            "context_09_positive_integers"].ask_for_intent(
                                str_intent_prompt = None,
                                str_intent_feature = strSelectedLexicalEntryIndex,
                                lst_str_expected_intents = [str(i+1) for i in range(int_lexical_entries_count)])
                if strSelectedLexicalEntryIndex == "" :
                    intSelectedLexicalEntryIndex = random.randint(
                        0, int_lexical_entries_count - 1)
                    str_lexical_entry_identifier = \
                        chat_state.current_synset.list_of_lexical_entries_identifiers[
                            intSelectedLexicalEntryIndex]
                    updated_LexicalEntry = lexicon.dictionary_of_lexical_entries[
                        str_lexical_entry_identifier]
                    boolIsSelectedLexicalEntryIndexValid = True
                elif strSelectedLexicalEntryIndex == "no" :
                    updated_LexicalEntry = None
                    boolIsSelectedLexicalEntryIndexValid = True
                    print(Fore.GREEN +
                          "No problem. We can always explore other things ..." +
                          Style.RESET_ALL)
                elif strSelectedLexicalEntryIndex.isdigit() :
                    intSelectedLexicalEntryIndex = int(strSelectedLexicalEntryIndex) - 1
                    if 0 <= intSelectedLexicalEntryIndex < int_lexical_entries_count :
                        str_lexical_entry_identifier = \
                            chat_state.current_synset.list_of_lexical_entries_identifiers[
                                intSelectedLexicalEntryIndex]
                        updated_LexicalEntry = lexicon.dictionary_of_lexical_entries[
                            str_lexical_entry_identifier]
                        boolIsSelectedLexicalEntryIndexValid = True
                        print(Fore.GREEN +
                              'Sounds good. Let\'s discuss word # {0}, i.e. the {1} "{2}".'.format(
                                  intSelectedLexicalEntryIndex + 1,
                                  dict_parts_of_speech_code_to_name[updated_LexicalEntry.part_of_speech],
                                  updated_LexicalEntry.written_form,
                                  ) + Style.RESET_ALL)
                    else :
                        print(
                            Fore.RED +
                            "I expected the word index to be from 1 to {0}.".format(
                                int_lexical_entries_count)  + Style.RESET_ALL)
                else :
                    print(Fore.RED +
                        'I do not get it. I expected either silence, or a valid integer, or just "no" from you.' +
                        Style.RESET_ALL)
            if updated_LexicalEntry is not None and \
               chat_state.current_lexical_entry != updated_LexicalEntry :
                chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                chat_state.current_lexical_entry = updated_LexicalEntry
    else :
        updated_LexicalEntry = None
        print(
            Fore.RED +
            'The "{0}" form of the word "{1}" makes no sense!'.format(
            dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,) + Style.RESET_ALL)

    return updated_LexicalEntry


def process_found_sense(
        found_Sense, lexicon, chat_state, dict_all_context_names_to_contexts) :

    updated_Sense = None

    if chat_state.current_sense != found_Sense :
        chat_state.previous_sense = chat_state.current_sense
        chat_state.current_sense = found_Sense

    intNumSenseRelations = len(
        chat_state.current_sense.dictionary_of_sense_relations)
    if intNumSenseRelations > 0 :
        lst_candidate_sense_relations = list(
            chat_state.current_sense.dictionary_of_sense_relations.values())

        # Listing unique pairs {str_sense_relation_type, str_sense_relation_desc}.
        print(
            Fore.CYAN +
            'This sense of the {0} "{1}" has the following types of sense relations:'.format(
            dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            ) + Style.RESET_ALL)
        set_tpl_str_sense_relation_type_desc = set()
        for tmp_sense_relation_obj in lst_candidate_sense_relations :
            if tmp_sense_relation_obj.sense_relation_type != "other" :
                str_sense_relation_type = tmp_sense_relation_obj.sense_relation_type
                str_sense_relation_desc = dict_relations_between_senses_name_to_descr[str_sense_relation_type]
            else :
                str_sense_relation_type = tmp_sense_relation_obj.sense_relation_subtype
                str_sense_relation_desc = dict_other_relations_between_senses_name_to_descr[str_sense_relation_type]
            set_tpl_str_sense_relation_type_desc.add((str_sense_relation_type, str_sense_relation_desc))
        for (i, (str_sense_relation_type, str_sense_relation_desc)) in enumerate(set_tpl_str_sense_relation_type_desc) :
            print(Fore.CYAN + '{0}. "{1}": {2}.'.format(
                i + 1,
                str_sense_relation_type.replace("_"," "),
                str_sense_relation_desc,) + Style.RESET_ALL)

        print(
            Fore.CYAN +
            'This sense of the {0} "{1}" has the following sense relations:'.format(
            dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            ) + Style.RESET_ALL)
        lst_str_candidate_sense_relations_types = []
        for (i, tmp_sense_relation_obj) in enumerate(lst_candidate_sense_relations) :
            if tmp_sense_relation_obj.sense_relation_type != "other" :
                str_sense_relation_type = tmp_sense_relation_obj.sense_relation_type
                lst_str_candidate_sense_relations_types.append(
                    tmp_sense_relation_obj.sense_relation_type)
            else :
                str_sense_relation_type = tmp_sense_relation_obj.sense_relation_subtype
                lst_str_candidate_sense_relations_types.append(
                    tmp_sense_relation_obj.sense_relation_type + "-" +
                    tmp_sense_relation_obj.sense_relation_subtype)
            tmp_target_sense_obj = lexicon.dictionary_of_senses[tmp_sense_relation_obj.target_sense_identifier]
            tmp_target_synset_obj = lexicon.dictionary_of_synsets[tmp_target_sense_obj.synset_identifier]
            lst_lex_ent_ids =  tmp_target_synset_obj.list_of_lexical_entries_identifiers
            if len(lst_lex_ent_ids) > 0 :
                str_synset_lexical_entries = ': ' + ', '.join([
                    lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form +
                    " (" + dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].part_of_speech] + ")"
                    for str_lexical_entry_identifier in lst_lex_ent_ids]) + ''
            else :
                str_synset_lexical_entries = ""
            print(Fore.CYAN + '* "{0}"{1}.'.format(
                str_sense_relation_type.replace("_"," "),
                str_synset_lexical_entries,) + Style.RESET_ALL)
        print(
            Fore.MAGENTA +
            'Which sense relation are you interested in?' +
            '\nPlease, keep silence, if you are not interested in any of them.' +
            Style.RESET_ALL)
        str_intent_feature = input().strip()
        if str_intent_feature != "" :
            str_selected_sense_relation_code = \
                dict_all_context_names_to_contexts[
                    "context_02_sense_relation_type"].ask_for_intent(
                str_intent_prompt = None,
                str_intent_feature = str_intent_feature,
                lst_str_expected_intents = lst_str_candidate_sense_relations_types)
            if str_selected_sense_relation_code.startswith("other") :
                str_selected_sense_relation_name = str_selected_sense_relation_code.split("-")[-1]
                str_sense_relation_descr = "{0}".format(
                    dict_other_relations_between_senses_name_to_descr[
                        str_selected_sense_relation_name])
            else :
                str_selected_sense_relation_name = str_selected_sense_relation_code
                str_sense_relation_descr = "{0}".format(
                    dict_relations_between_senses_name_to_descr[
                        str_selected_sense_relation_code])
            print(
                Fore.GREEN +
                'Got it! Let us talk about another sense of the {0} "{1}", which has a type "{2}" relation with its current sense.'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    str_selected_sense_relation_name.replace("_"," ")) + Style.RESET_ALL)
            print(Fore.CYAN + 'The sense relation "{0}" is defined as follows: "{1}".'.format(
                str_selected_sense_relation_name.replace("_"," "),
                str_sense_relation_descr) +
                Style.RESET_ALL)
            lst_pre_selected_candidate_sense_relations = []
            for tmp_sense_relation_obj in lst_candidate_sense_relations :
                if tmp_sense_relation_obj.sense_relation_type != "other" :
                    str_sense_relation_type = tmp_sense_relation_obj.sense_relation_type
                else :
                    str_sense_relation_type = tmp_sense_relation_obj.sense_relation_type + "-" + \
                        tmp_sense_relation_obj.sense_relation_subtype
                if str_sense_relation_type == str_selected_sense_relation_code :
                    lst_pre_selected_candidate_sense_relations.append(tmp_sense_relation_obj)
            intNumSenseRelations = len(lst_pre_selected_candidate_sense_relations)
            if intNumSenseRelations > 1 :
                print(Fore.CYAN + 'These are the sense relations of type "{0}":'.format(
                    str_selected_sense_relation_code,) + Style.RESET_ALL)
                for (i, tmp_sense_relation_obj) in enumerate(lst_pre_selected_candidate_sense_relations) :
                    if tmp_sense_relation_obj.sense_relation_type != "other" :
                        str_sense_relation_type = tmp_sense_relation_obj.sense_relation_type
                        lst_str_candidate_sense_relations_types.append(
                            tmp_sense_relation_obj.sense_relation_type)
                    else :
                        str_sense_relation_type = tmp_sense_relation_obj.sense_relation_subtype
                        lst_str_candidate_sense_relations_types.append(
                            tmp_sense_relation_obj.sense_relation_type + "-" +
                            tmp_sense_relation_obj.sense_relation_subtype)
                    tmp_target_sense_obj = lexicon.dictionary_of_senses[
                        tmp_sense_relation_obj.target_sense_identifier]
                    tmp_target_synset_obj = lexicon.dictionary_of_synsets[
                        tmp_target_sense_obj.synset_identifier]
                    lst_target_lex_ent_ids =  tmp_target_synset_obj.list_of_lexical_entries_identifiers
                    if len(lst_target_lex_ent_ids) > 0 :
                        str_synset_lexical_entries = ': ' + ', '.join([
                            lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form +
                            " (" + dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[
                                str_lexical_entry_identifier].part_of_speech] + ")"
                            for str_lexical_entry_identifier in lst_target_lex_ent_ids]) + ''
                    else :
                        str_synset_lexical_entries = ""
                    print(Fore.CYAN + '{0}. "{1}"{2}.'.format(
                        i + 1, str_sense_relation_type.replace("_"," "), str_synset_lexical_entries,) + Style.RESET_ALL)
                boolIsSelectedSenseRelationIndexValid = False
                while not boolIsSelectedSenseRelationIndexValid :
                    print(
                        Fore.MAGENTA +
                        "Which sense relation are you interested in, from 1 to {0}?".format(intNumSenseRelations) +
                        "\nYou can stay silent if you want me to choose a sense relation." +
                        Style.RESET_ALL)
                    strSelectedSenseRelationIndex = input().strip()
                    if strSelectedSenseRelationIndex != "" :
                        strSelectedSenseRelationIndex = \
                            dict_all_context_names_to_contexts[
                                "context_09_positive_integers"].ask_for_intent(
                                    str_intent_prompt = None,
                                    str_intent_feature = strSelectedSenseRelationIndex,
                                    lst_str_expected_intents = [str(i+1) for i in range(intNumSenseRelations)])
                    if strSelectedSenseRelationIndex == "" :
                        intSelectedSenseRelationIndex = random.randint(0, intNumSenseRelations - 1)
                        boolIsSelectedSenseRelationIndexValid = True
                    elif strSelectedSenseRelationIndex.isdigit() :
                        intSelectedSenseRelationIndex = int(strSelectedSenseRelationIndex) - 1
                        if 0 <= intSelectedSenseRelationIndex < intNumSenseRelations :
                            boolIsSelectedSenseRelationIndexValid = True
                        else :
                            print(
                                Fore.RED +
                                "I expected the sense relation index to be from 1 to {0}.".format(
                                intNumSenseRelations) + Style.RESET_ALL)
                    else :
                        print(
                            Fore.RED +
                            "I do not get it. I expected either an empty response or a valid integer." +
                            Style.RESET_ALL)
            else :
                intSelectedSenseRelationIndex = 0

            selected_sense_relation_obj = lst_pre_selected_candidate_sense_relations[
                intSelectedSenseRelationIndex]
            selected_target_sense_obj = lexicon.dictionary_of_senses[
                selected_sense_relation_obj.target_sense_identifier]
            selected_target_synset_obj = lexicon.dictionary_of_synsets[
                selected_target_sense_obj.synset_identifier]
            lst_selected_target_lex_ent_ids = selected_target_synset_obj.list_of_lexical_entries_identifiers
            if selected_sense_relation_obj.sense_relation_type == "other" :
                str_selected_sense_relation_name = selected_sense_relation_obj.sense_relation_subtype
            else :
                str_selected_sense_relation_name = selected_sense_relation_obj.sense_relation_type

            if len(lst_selected_target_lex_ent_ids) > 0 :
                print(Fore.CYAN + 'The {0} "{1}" has the sense relation of type "{2}" with the following lexical entries:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    str_selected_sense_relation_name.replace("_"," "),
                    ) + Style.RESET_ALL)
                for (i, str_lexical_entry_identifier) in enumerate(lst_selected_target_lex_ent_ids) :
                    print(Fore.CYAN + '{0}. "{1}" ({2}).'.format(
                        i + 1,
                        lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form,
                        dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[
                            str_lexical_entry_identifier].part_of_speech],
                        ) + Style.RESET_ALL)
                boolIsSelectedLexicalEntryIndexValid = False
                intNumLexicalEntries = len(lst_selected_target_lex_ent_ids)
                while not boolIsSelectedLexicalEntryIndexValid :
                    print(
                        Fore.MAGENTA +
                        "Which lexical entries are you interested in, from 1 to {0}?".format(intNumLexicalEntries) +
                        "\nYou can stay silent if you want me to choose a lexical entry." +
                        Style.RESET_ALL)
                    strSelectedLexicalEntryIndex = input().strip()
                    if strSelectedLexicalEntryIndex != "" :
                        strSelectedLexicalEntryIndex = \
                            dict_all_context_names_to_contexts[
                                "context_09_positive_integers"].ask_for_intent(
                                    str_intent_prompt = None,
                                    str_intent_feature = strSelectedLexicalEntryIndex,
                                    lst_str_expected_intents = [str(i+1) for i in range(intNumLexicalEntries)])
                    if strSelectedLexicalEntryIndex == "" :
                        intSelectedLexicalEntryIndex = random.randint(0, intNumLexicalEntries - 1)
                        boolIsSelectedLexicalEntryIndexValid = True
                    elif strSelectedLexicalEntryIndex.isdigit() :
                        intSelectedLexicalEntryIndex = int(strSelectedLexicalEntryIndex) - 1
                        if 0 <= intSelectedLexicalEntryIndex < intNumLexicalEntries :
                            boolIsSelectedLexicalEntryIndexValid = True
                        else :
                            print(
                                Fore.RED +
                                "I expected the lexical entry index to be from 1 to {0}.".format(
                                intNumLexicalEntries) + Style.RESET_ALL)
                    else :
                        print(
                            Fore.RED +
                            "I do not get it. I expected either an empty response or a valid integer." +
                            Style.RESET_ALL)
                str_selected_lexical_entry_id = lst_selected_target_lex_ent_ids[intSelectedLexicalEntryIndex]
                selected_lexical_entry_obj = lexicon.dictionary_of_lexical_entries[str_selected_lexical_entry_id]
                print(
                    Fore.GREEN +
                    'I got it. We will focus on the lexical entry # {0}, i. e. the {1} "{2}".'.format(
                    intSelectedLexicalEntryIndex + 1,
                    dict_parts_of_speech_code_to_name[selected_lexical_entry_obj.part_of_speech],
                    selected_lexical_entry_obj.written_form,
                    ) + Style.RESET_ALL)

                chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                chat_state.current_lexical_entry = selected_lexical_entry_obj
                chat_state.previous_sense = chat_state.current_sense
                chat_state.current_sense = selected_target_sense_obj
                chat_state.previous_synset = chat_state.current_synset
                chat_state.current_synset = selected_target_synset_obj
                updated_Sense = chat_state.current_sense
            else :
                print(Fore.RED + 'The {0} "{1}" has no sense relation of type {2} with any word!'  + Style.RESET_ALL)
        else :
            print(
                Fore.GREEN +
                "Then let's discuss something else, beyond related senses ..." +
                Style.RESET_ALL)

    return updated_Sense


def process_found_synset(
        found_Synset, lexicon, chat_state, dict_all_context_names_to_contexts) :

    updated_Synset = None

    if chat_state.current_synset != found_Synset :
        chat_state.previous_synset = chat_state.current_synset
        chat_state.current_synset = found_Synset

    intNumSynsetRelations = len(
        chat_state.current_synset.dictionary_of_synset_relations)
    if intNumSynsetRelations > 0 :
        lst_candidate_synset_relations = list(
            chat_state.current_synset.dictionary_of_synset_relations.values())

        # Listing unique pairs {str_synset_relation_type, str_synset_relation_desc}.
        print(
            Fore.CYAN +
            'This synonym set of the {0} "{1}" has the following types of relations with other synonym sets:'.format(
            dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            ) + Style.RESET_ALL)
        set_tpl_str_synset_relation_type_desc = set()
        for tmp_synset_relation_obj in lst_candidate_synset_relations :
            str_synset_relation_type = tmp_synset_relation_obj.synset_relation_type
            str_synset_relation_desc = dict_relations_between_synsets_name_to_descr[str_synset_relation_type]
            set_tpl_str_synset_relation_type_desc.add((str_synset_relation_type, str_synset_relation_desc))
        for (i, (str_synset_relation_type, str_synset_relation_desc)) in enumerate(set_tpl_str_synset_relation_type_desc) :
            print(Fore.CYAN + '{0}. "{1}": {2}.'.format(
                i + 1,
                str_synset_relation_type.replace("_"," "),
                str_synset_relation_desc,) + Style.RESET_ALL)

        print(
            Fore.CYAN +
            'This synonym set of the {0} "{1}" has the following relations with other synonym sets:'.format(
            dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
            chat_state.current_lexical_entry.written_form,
            ) + Style.RESET_ALL)
        lst_str_candidate_synset_relations_types = []
        for (i, tmp_synset_relation_obj) in enumerate(lst_candidate_synset_relations) :
            str_synset_relation_type = tmp_synset_relation_obj.synset_relation_type
            str_synset_relation_desc = dict_relations_between_synsets_name_to_descr[str_synset_relation_type]
            lst_str_candidate_synset_relations_types.append(
                tmp_synset_relation_obj.synset_relation_type)
            tmp_target_synset_obj = lexicon.dictionary_of_synsets[tmp_synset_relation_obj.target_synset_identifier]
            lst_lex_ent_ids =  tmp_target_synset_obj.list_of_lexical_entries_identifiers
            if len(lst_lex_ent_ids) > 0 :
                str_synset_lexical_entries = ': ' + ', '.join([
                    lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form +
                    " (" + dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].part_of_speech] + ")"
                    for str_lexical_entry_identifier in lst_lex_ent_ids]) + ''
            else :
                str_synset_lexical_entries = ""
            print(Fore.CYAN + '* "{0}"{1}.'.format(
                str_synset_relation_type.replace("_"," "),
                str_synset_lexical_entries,) + Style.RESET_ALL)
        print(
            Fore.MAGENTA +
            'Which synonym set relation are you interested in?' +
            Style.RESET_ALL)
        print(
            Fore.MAGENTA +
            'Please, stay silent, if you are not interested in any of them.' +
            Style.RESET_ALL)
        str_intent_feature = input().strip()
        if str_intent_feature != "" :
            str_selected_synset_relation_code = \
                dict_all_context_names_to_contexts[
                    "context_03_synset_relation_type"].ask_for_intent(
                str_intent_prompt = None,
                str_intent_feature = str_intent_feature,
                lst_str_expected_intents = lst_str_candidate_synset_relations_types)
            str_selected_synset_relation_name = str_selected_synset_relation_code
            str_synset_relation_descr = "{0}".format(
                dict_relations_between_synsets_name_to_descr[
                    str_selected_synset_relation_code])
            print(
                Fore.GREEN +
                'Got it! Let us talk about another synonym set, which has "{0}" relation to the synonym set of the {1} "{2}".'.format(
                    str_selected_synset_relation_name.replace("_", " "),
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    ) + Style.RESET_ALL)
            print(Fore.CYAN + 'The synonym set relation "{0}" is defined as follows: "{1}".'.format(
                str_selected_synset_relation_name.replace("_", " "),
                str_synset_relation_descr) + Style.RESET_ALL)
            lst_pre_selected_candidate_synset_relations = []
            for tmp_synset_relation_obj in lst_candidate_synset_relations :
                str_synset_relation_type = tmp_synset_relation_obj.synset_relation_type
                if str_synset_relation_type == str_selected_synset_relation_code :
                    lst_pre_selected_candidate_synset_relations.append(tmp_synset_relation_obj)
            intNumSynsetRelations = len(lst_pre_selected_candidate_synset_relations)
            if intNumSynsetRelations > 1 :
                print(Fore.CYAN + 'These are the synonym set relations of type "{0}":'.format(
                    str_selected_synset_relation_code.replace("_", " "),) + Style.RESET_ALL)
                for (i, tmp_synset_relation_obj) in enumerate(lst_pre_selected_candidate_synset_relations) :
                    str_synset_relation_type = tmp_synset_relation_obj.synset_relation_type
                    lst_str_candidate_synset_relations_types.append(
                        tmp_synset_relation_obj.synset_relation_type)
                    tmp_target_synset_obj = lexicon.dictionary_of_synsets[
                        tmp_synset_relation_obj.target_synset_identifier]
                    lst_target_lex_ent_ids =  tmp_target_synset_obj.list_of_lexical_entries_identifiers
                    if len(lst_target_lex_ent_ids) > 0 :
                        str_synset_lexical_entries = ': ' + ', '.join([
                            lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form +
                            " (" + dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[
                                str_lexical_entry_identifier].part_of_speech] + ")"
                            for str_lexical_entry_identifier in lst_target_lex_ent_ids]) + ''
                    else :
                        str_synset_lexical_entries = ""
                    print(Fore.CYAN + '{0}. "{1}"{2}.'.format(
                        i + 1, str_synset_relation_type.replace("_"," "), str_synset_lexical_entries,) + Style.RESET_ALL)
                boolIsSelectedSynsetRelationIndexValid = False
                while not boolIsSelectedSynsetRelationIndexValid :
                    print(
                        Fore.MAGENTA +
                        "Which synonym set relation are you interested in, from 1 to {0}?".format(intNumSynsetRelations) +
                        "\nYou can stay silent if you want me to choose a synonym set relation." +
                        Style.RESET_ALL)
                    strSelectedSynsetRelationIndex = input().strip()
                    if strSelectedSynsetRelationIndex != "" :
                        strSelectedSynsetRelationIndex = \
                            dict_all_context_names_to_contexts[
                                "context_09_positive_integers"].ask_for_intent(
                                    str_intent_prompt = None,
                                    str_intent_feature = strSelectedSynsetRelationIndex,
                                    lst_str_expected_intents = [str(i+1) for i in range(intNumSynsetRelations)])
                    if strSelectedSynsetRelationIndex == "" :
                        intSelectedSynsetRelationIndex = random.randint(0, intNumSynsetRelations - 1)
                        boolIsSelectedSynsetRelationIndexValid = True
                    elif strSelectedSynsetRelationIndex.isdigit() :
                        intSelectedSynsetRelationIndex = int(strSelectedSynsetRelationIndex) - 1
                        if 0 <= intSelectedSynsetRelationIndex < intNumSynsetRelations :
                            boolIsSelectedSynsetRelationIndexValid = True
                        else :
                            print(
                                Fore.RED +
                                "I expected the synonym set relation index to be from 1 to {0}.".format(
                                intNumSynsetRelations) + Style.RESET_ALL)
                    else :
                        print(
                            Fore.RED +
                            "I do not get it. I expected either an empty response or a valid integer." +
                            Style.RESET_ALL)
            else :
                intSelectedSynsetRelationIndex = 0

            selected_synset_relation_obj = lst_pre_selected_candidate_synset_relations[
                intSelectedSynsetRelationIndex]
            selected_target_synset_obj = lexicon.dictionary_of_synsets[
                selected_synset_relation_obj.target_synset_identifier]
            lst_selected_target_lex_ent_ids = selected_target_synset_obj.list_of_lexical_entries_identifiers
            str_selected_synset_relation_name = selected_synset_relation_obj.synset_relation_type

            if len(lst_selected_target_lex_ent_ids) > 0 :
                print(Fore.CYAN + 'The {0} "{1}" has the synonym set relation of type "{2}" with the following lexical entries:'.format(
                    dict_parts_of_speech_code_to_name[chat_state.current_lexical_entry.part_of_speech],
                    chat_state.current_lexical_entry.written_form,
                    str_selected_synset_relation_name.replace("_", " "),
                    ) + Style.RESET_ALL)
                for (i, str_lexical_entry_identifier) in enumerate(lst_selected_target_lex_ent_ids) :
                    print(Fore.CYAN + '{0}. "{1}" ({2}).'.format(
                        i + 1,
                        lexicon.dictionary_of_lexical_entries[str_lexical_entry_identifier].written_form,
                        dict_parts_of_speech_code_to_name[lexicon.dictionary_of_lexical_entries[
                            str_lexical_entry_identifier].part_of_speech],
                        ) + Style.RESET_ALL)
                boolIsSelectedLexicalEntryIndexValid = False
                intNumLexicalEntries = len(lst_selected_target_lex_ent_ids)
                if intNumLexicalEntries > 1 :
                    while not boolIsSelectedLexicalEntryIndexValid :
                        print(
                            Fore.MAGENTA +
                            "Which lexical entries are you interested in, from 1 to {0}?".format(intNumLexicalEntries) +
                            "\nYou can stay silent if you want me to choose a lexical entry." +
                            Style.RESET_ALL)
                        strSelectedLexicalEntryIndex = input().strip()
                        if strSelectedLexicalEntryIndex != "" :
                            strSelectedLexicalEntryIndex = \
                                dict_all_context_names_to_contexts[
                                    "context_09_positive_integers"].ask_for_intent(
                                        str_intent_prompt = None,
                                        str_intent_feature = strSelectedLexicalEntryIndex,
                                        lst_str_expected_intents = [str(i+1) for i in range(intNumLexicalEntries)])
                        if strSelectedLexicalEntryIndex == "" :
                            intSelectedLexicalEntryIndex = random.randint(0, intNumLexicalEntries - 1)
                            boolIsSelectedLexicalEntryIndexValid = True
                        elif strSelectedLexicalEntryIndex.isdigit() :
                            intSelectedLexicalEntryIndex = int(strSelectedLexicalEntryIndex) - 1
                            if 0 <= intSelectedLexicalEntryIndex < intNumLexicalEntries :
                                boolIsSelectedLexicalEntryIndexValid = True
                            else :
                                print(
                                    Fore.RED +
                                    "I expected the lexical entry index to be from 1 to {0}.".format(
                                    intNumLexicalEntries) + Style.RESET_ALL)
                        else :
                            print(
                                Fore.RED +
                                "I do not get it. I expected either an empty response or a valid integer." +
                                Style.RESET_ALL)
                else :
                    intSelectedLexicalEntryIndex = 0
                str_selected_lexical_entry_id = lst_selected_target_lex_ent_ids[intSelectedLexicalEntryIndex]
                selected_lexical_entry_obj = lexicon.dictionary_of_lexical_entries[str_selected_lexical_entry_id]
                print(
                    Fore.GREEN +
                    'Great! We will focus on the lexical entry # {0}, i. e. the {1} "{2}".'.format(
                    intSelectedLexicalEntryIndex + 1,
                    dict_parts_of_speech_code_to_name[selected_lexical_entry_obj.part_of_speech],
                    selected_lexical_entry_obj.written_form,
                    ) + Style.RESET_ALL)

                lst_senses_for_curr_lexical_entry = list(
                    selected_lexical_entry_obj.dictionary_of_senses.values())
                lst_relevant_senses_for_curr_lexical_entry = []
                for (i, tmp_Sense) in enumerate(lst_senses_for_curr_lexical_entry) :
                    tmp_Synset = lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier]
                    if tmp_Sense.synset_identifier == selected_target_synset_obj.identifier and \
                       tmp_Synset == selected_target_synset_obj :
                           lst_relevant_senses_for_curr_lexical_entry.append(tmp_Sense)
                del lst_senses_for_curr_lexical_entry
                intNumRelevantSenses = len(lst_relevant_senses_for_curr_lexical_entry)
                if intNumRelevantSenses >= 1 :
                    if intNumRelevantSenses == 1 :
                        intSelectedSenseIndex = 0
                        tmp_Sense = lst_relevant_senses_for_curr_lexical_entry[intSelectedSenseIndex]
                        tmp_Synset = lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier]
                        str_definitions = '"' + '", or "'.join(
                            [definition.text for definition in selected_target_synset_obj.list_of_definitions]) + '"'
                        if str_definitions == '""' :
                            str_definitions = "no definition"
                        print(
                            Fore.CYAN +
                            'The {0} "{1}" has just one synonym set-relevant sense, such as {2}.'.format(
                            dict_parts_of_speech_code_to_name[
                                selected_lexical_entry_obj.part_of_speech],
                                selected_lexical_entry_obj.written_form,
                                str_definitions,) + Style.RESET_ALL)
                        selected_target_sense_obj = tmp_Sense
                    elif intNumRelevantSenses > 1 :
                        print(
                            Fore.CYAN +
                            'The {0} "{1}" has {2} synonym set-relevant senses:'.format(
                            dict_parts_of_speech_code_to_name[
                                selected_lexical_entry_obj.part_of_speech],
                            selected_lexical_entry_obj.written_form,
                            intNumRelevantSenses) + Style.RESET_ALL)
                        for (i, tmp_Sense) in enumerate(lst_relevant_senses_for_curr_lexical_entry) :
                            tmp_Synset = lexicon.dictionary_of_synsets[tmp_Sense.synset_identifier]
                            str_definitions = '"' + '", or "'.join(
                                [definition_obj.text for definition_obj in tmp_Synset.list_of_definitions]) + '"'
                            if str_definitions == '""' :
                                str_definitions = "no definition"
                            print(Fore.CYAN +'{0}. {1}.'.format(
                                i + 1, str_definitions) + Style.RESET_ALL)
                        boolIsSelectedSenseIndexValid = False
                        while not boolIsSelectedSenseIndexValid :
                            print(
                                Fore.MAGENTA +
                                "Which sense are you interested in, from 1 to {0}?".format(intNumRelevantSenses) +
                                "\nYou can stay silent if you are interested in any random relevant sense." +
                                Style.RESET_ALL)
                            strSelectedSenseIndex = input().strip()
                            if strSelectedSenseIndex != "" :
                                strSelectedSenseIndex = \
                                    dict_all_context_names_to_contexts[
                                        "context_09_positive_integers"].ask_for_intent(
                                            str_intent_prompt = None,
                                            str_intent_feature = strSelectedSenseIndex,
                                            lst_str_expected_intents = [str(i+1) for i in range(intNumRelevantSenses)])
                            if strSelectedSenseIndex == "" :
                                intSelectedSenseIndex = random.randint(0, intNumRelevantSenses - 1)
                                boolIsSelectedSenseIndexValid = True
                            elif strSelectedSenseIndex.isdigit() :
                                intSelectedSenseIndex = int(strSelectedSenseIndex) - 1
                                if 0 <= intSelectedSenseIndex < intNumRelevantSenses :
                                    boolIsSelectedSenseIndexValid = True
                                else :
                                    print(
                                        Fore.RED +
                                        "I expected the relevant sense index to be from 1 to {0}.".format(
                                            intNumRelevantSenses)  + Style.RESET_ALL)
                            else :
                                print(
                                    Fore.RED +
                                    "I do not get it. I expected either an empty response or a valid integer." +
                                    Style.RESET_ALL)
                        selected_target_sense_obj = lst_relevant_senses_for_curr_lexical_entry[intSelectedSenseIndex]
###############################################################################
                else :
                    selected_target_sense_obj = None
                    print(
                        Fore.RED +
                        'The "{0}" form of the word "{1}" has no sense that would be consistent with its synonym set!'.format(
                        dict_parts_of_speech_code_to_name[selected_lexical_entry_obj.part_of_speech],
                        selected_lexical_entry_obj.written_form,) + Style.RESET_ALL)

                chat_state.previous_lexical_entry = chat_state.current_lexical_entry
                chat_state.current_lexical_entry = selected_lexical_entry_obj
                chat_state.previous_sense = chat_state.current_sense
                chat_state.current_sense = selected_target_sense_obj
                chat_state.previous_synset = chat_state.current_synset
                chat_state.current_synset = selected_target_synset_obj
                updated_Synset = chat_state.current_synset
            else :
                print(
                    Fore.RED +
                    'The {0} "{1}" has no synonym set relation of type {2} with any word!' +
                    Style.RESET_ALL)
        else :
            print(
                Fore.GREEN +
                "Then let's discuss something else, beyond related sets of synonyms ..." +
                Style.RESET_ALL)

    return updated_Synset


###############################################################################


def main() -> int :

    (lexicon, chat_state, dict_all_context_names_to_contexts) = initialize_chat()

    bool_chat_end_intent = False
    while not bool_chat_end_intent :
        if chat_state.current_lexical_entry is None :
            while chat_state.current_lexical_entry is None :
                find_lexical_entry(
                    lexicon = lexicon, chat_state = chat_state,
                    dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)
        else :
            bool_chat_end_intent = ask_to_finish_chat(
                dict_all_context_names_to_contexts = dict_all_context_names_to_contexts)
            if not bool_chat_end_intent :
                bool_replace_lexical_entry = ask_whether_to_replace_lexical_entry(
                    chat_state, dict_all_context_names_to_contexts)
                if bool_replace_lexical_entry :
                    find_lexical_entry(
                        lexicon = lexicon, chat_state = chat_state,
                        dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)
                    while chat_state.current_lexical_entry is None :
                        find_lexical_entry(
                            lexicon = lexicon, chat_state = chat_state,
                            dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)
        if not bool_chat_end_intent :
            found_LexicalEntry = chat_state.current_lexical_entry
            while found_LexicalEntry is not None :
                found_LexicalEntry = process_found_lexical_entry(
                    found_LexicalEntry = found_LexicalEntry, lexicon = lexicon, chat_state = chat_state,
                    dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)
            found_Sense = chat_state.current_sense
            while found_Sense is not None :
                found_Sense = process_found_sense(
                    found_Sense = found_Sense, lexicon = lexicon, chat_state = chat_state,
                    dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)
            found_Synset = chat_state.current_synset
            while found_Synset is not None :
                found_Synset = process_found_synset(
                    found_Synset = found_Synset, lexicon = lexicon, chat_state = chat_state,
                    dict_all_context_names_to_contexts = dict_all_context_names_to_contexts,)

    return 0


if __name__ == '__main__':
    sys.exit(main())
