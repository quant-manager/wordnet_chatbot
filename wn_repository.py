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
# WordNet
# https://github.com/globalwordnet/English-Wordnet?tab=readme-ov-file
# https://wordnet.princeton.edu/
# https://globalwordnet.github.io/schemas/
# https://globalwordnet.github.io/schemas/dc/
# https://en-word.net/lemma/ECG#ECG-n
#
# Use the following to generate dialogs(!):
# https://globalwordnet.github.io/gwadoc/
#
# https://realpython.com/python-xml-parser/
# https://docs.python.org/3/library/xml.etree.elementtree.html
###############################################################################


import os
import sys
import xml.etree.ElementTree as ET
import pickle
import lzma

DIR_PATH_DATA_PKL_XZ_WORDNET = "data_pkl_xz_wordnet"
FILE_NAME_DATA_PKL_XZ_WORDNET = "lexicon_oewn"


# Sense Relation's counts by their Syntactic Behaviour types:
# 1. v: verb
# 2. t: transitive
#    i: intransitive
# 3. a :animate
#    i: inanimate
# 4. a :animate
#    i: inanimate
dict_sense_relations_syntactic_behaviours_code_to_template = \
{
    # <SyntacticBehaviour id="vtai" subcategorizationFrame="Somebody ----s something"/>
    "vtai" : "(Somebody ----s something)", # 13225
    "via" : "(Somebody ----s)", # 5148
    "vtaa" : "(Somebody ----s somebody)", # 5077
    "vtii" : "(Something ----s something)", # 4895
    "vii" : "(Something ----s)", # 3607
    "via-pp" : "(Somebody ----s PREPOSITIONAL PHRASE)", # 2215
    "vtia" : "(Something ----s somebody)", # 2127
    "vii-pp" : "(Something is ----ing PREPOSITIONAL PHRASE)", # 1052
    "vtai-pp" : "(Somebody ----s something PREPOSITIONAL PHRASE)", # 1023
    "via-that" : "(Somebody ----s that CLAUSE)", # 675
    "vtaa-pp" : "(Somebody ----s somebody PREPOSITIONAL PHRASE)", # 475
    "vtai-to" : "(Somebody ----s something to somebody)", # 386
    "ditransitive" : "(Somebody ----s somebody something)", # 202
    "vtaa-to-inf" : "(Somebody ----s somebody to INFINITIVE)", # 198
    "via-to-inf" : "(Somebody ----s to INFINITIVE)", # 190
    "vtai-from" : "(Somebody ----s something from somebody)", # 152
    "vtaa-with" : "(Somebody ----s somebody with something)", # 124
    "via-ger" : "(Somebody ----s VERB-ing)", # 111
    "vtai-with" : "(Somebody ----s something with something)", # 104
    "vii-adj" : "(Something ----s Adjective/Noun)", # 69
    "via-adj" : "(Somebody ----s Adjective)", # 67
    "via-to" : "(Somebody ----s to somebody)", # 64
    "vtaa-of" : "(Somebody ----s somebody of something)", # 57
    "via-on-inanim" : "(Somebody ----s on something)", # 57
    "vtii-adj" : "(Something ----s something Adjective/Noun)", # 54
    "via-whether-inf" : "(Somebody ----s whether INFINITIVE)", # 51
    "nonreferential" : "(It is ----ing)", # 43
    "nonreferential-sent" : "(It ----s that CLAUSE)", # 42
    "vtaa-into-ger" : "(Somebody ----s somebody into V-ing something)", # 37
    "vibody" : "(Somebody's (body part) ----s)", # 35
    "vtai-on" : "(Somebody ----s something on somebody)", # 34
    "vii-to" : "(Something ----s to somebody)", # 25
    "vtaa-inf" : "(Somebody ----s somebody INFINITIVE)", # 19
    "via-inf" : "(Somebody ----s INFINITIVE)", # 9
    "vii-inf" : "(Something ----s INFINITIVE)", # 5
    "via-for" : "(Somebody ----s for something)", # 1
    "via-on-anim" : "(Somebody ----s on somebody)", # 1
    "via-at" : "(Somebody ----s at something)", # 1
    "via-out-of" : "(Somebody ----s out of somebody)", # 1
}

dict_sense_relations_syntactic_behaviours_code_to_descr = \
{
    # <SyntacticBehaviour id="vtai" subcategorizationFrame="Somebody ----s something"/>
    #
    # "vtai" : "(Somebody ----s something)", # 13225
    "vtai" : "<animate subject> <TRANSITIVE VERB> <inanimate object>", # 13225
    # "via" : "(Somebody ----s)", # 5148
    "via" : "<animate subject> <INTRANSITIVE VERB>", # 5148
    # "vtaa" : "(Somebody ----s somebody)", # 5077
    "vtaa" : "<animate subject> <TRANSITIVE VERB> <animate object>", # 5077
    # "vtii" : "(Something ----s something)", # 4895
    "vtii" : "<inanimate subject> <TRANSITIVE VERB> <inanimate object>", # 4895
    # "vii" : "(Something ----s)", # 3607
    "vii" : "<inanimate subject> <INTRANSITIVE VERB>", # 3607
    # "via-pp" : "(Somebody ----s PREPOSITIONAL PHRASE)", # 2215
    "via-pp" : "<animate subject> <INTRANSITIVE VERB> <prepositional phrase>", # 2215
    # "vtia" : "(Something ----s somebody)", # 2127
    "vtia" : "<inanimate subject> <TRANSITIVE VERB> <animate object>", # 2127
    # "vii-pp" : "(Something is ----ing PREPOSITIONAL PHRASE)", # 1052
    "vii-pp" : "<inanimate subject> am/is/are <PRESENT PARTICIPLE OF INTRANSITIVE VERB> <prepositional phrase>", # 1052
    # "vtai-pp" : "(Somebody ----s something PREPOSITIONAL PHRASE)", # 1023
    "vtai-pp" : "<animate subject> <TRANSITIVE VERB> <inanimate object> <prepositional phrase>", # 1023
    # "via-that" : "(Somebody ----s that CLAUSE)", # 675
    "via-that" : "<animate subject> <INTRANSITIVE VERB> that <clause>", # 675
    # "vtaa-pp" : "(Somebody ----s somebody PREPOSITIONAL PHRASE)", # 475
    "vtaa-pp" : "<animate subject> <TRANSITIVE VERB> <animate object> <prepositional phrase>", # 475
    # "vtai-to" : "(Somebody ----s something to somebody)", # 386
    "vtai-to" : "<animate subject> <TRANSITIVE VERB> <inanimate direct object> <animate indirect object>", # 386
    # "ditransitive" : "(Somebody ----s somebody something)", # 202
    "ditransitive" : "<animate subject> <DITRANSITIVE VERB> <animate indirect object> <inanimate direct object>", # 202
    # "vtaa-to-inf" : "(Somebody ----s somebody to INFINITIVE)", # 198
    "vtaa-to-inf" : "<animate subject> <TRANSITIVE VERB> <animate object> to <verb's infinitive>", # 198
    # "via-to-inf" : "(Somebody ----s to INFINITIVE)", # 190
    "via-to-inf" : "<animate subject> <INTRANSITIVE VERB> to <verb's infinitive>", # 190
    # "vtai-from" : "(Somebody ----s something from somebody)", # 152
    "vtai-from" : "<animate subject> <TRANSITIVE VERB> <inanimate direct object> from <animate indirect object>", # 152
    # "vtaa-with" : "(Somebody ----s somebody with something)", # 124
    "vtaa-with" : "<animate subject> <TRANSITIVE VERB> <animate direct object> with <inanimate indirect object>", # 124
    # "via-ger" : "(Somebody ----s VERB-ing)", # 111
    "via-ger" : "<animate subject> <INTRANSITIVE VERB> <verb's gerund>", # 111
    # "vtai-with" : "(Somebody ----s something with something)", # 104
    "vtai-with" : "<animate subject> <TRANSITIVE VERB> <inanimate direct object> with <inanimate indirect object> ...", # 104
    # "vii-adj" : "(Something ----s Adjective/Noun)", # 69
    "vii-adj" : "<inanimate subject> <INTRANSITIVE VERB> <adjective or noun>", # 69
    # "via-adj" : "(Somebody ----s Adjective)", # 67
    "via-adj" : "<animate subject> <INTRANSITIVE VERB> <adjective>", # 67
    # "via-to" : "(Somebody ----s to somebody)", # 64
    "via-to" : "<animate subject> <INTRANSITIVE VERB> to <animate noun>", # 64
    # "vtaa-of" : "(Somebody ----s somebody of something)", # 57
    "vtaa-of" : "<animate subject> <TRANSITIVE VERB> <animate object> of <inanimate noun>", # 57
    # "via-on-inanim" : "(Somebody ----s on something)", # 57
    "via-on-inanim" : "<animate subject> <INTRANSITIVE VERB> on <inanimate noun>", # 57
    # "vtii-adj" : "(Something ----s something Adjective/Noun)", # 54
    "vtii-adj" : "<inanimate subject> <TRANSITIVE VERB> <inanimate object> <adjective or noun>", # 54
    # "via-whether-inf" : "(Somebody ----s whether INFINITIVE)", # 51
    "via-whether-inf" : "<animate subject> <INTRANSITIVE VERB> whether <verb's infinitive>", # 51
    # "nonreferential" : "(It is ----ing)", # 43
    "nonreferential" : "it is/was/will <PRESENT PARTICIPLE OF NONREFERENTIAL VERB>", # 43
    # "nonreferential-sent" : "(It ----s that CLAUSE)", # 42
    "nonreferential-sent" : "it <NONREFERENTIAL VERB> that <clause>", # 42
    # "vtaa-into-ger" : "(Somebody ----s somebody into V-ing something)", # 37
    "vtaa-into-ger" : "<animate subject> <TRANSITIVE VERB> <animate direct object> into <verb's gerund> <inanimate indirect object>", # 37
    # "vibody" : "(Somebody's (body part) ----s)", # 35
    "vibody" : "<animate subject> <subject's body part> <INTRANSITIVE VERB>", # 35
    # "vtai-on" : "(Somebody ----s something on somebody)", # 34
    "vtai-on" : "<animate subject> <TRANSITIVE VERB> <inanimale direct object> on <animate indirect object>", # 34
    # "vii-to" : "(Something ----s to somebody)", # 25
    "vii-to" : "<inanimate subject> <INTRANSITIVE VERB> to <animate object>", # 25
    # "vtaa-inf" : "(Somebody ----s somebody INFINITIVE)", # 19
    "vtaa-inf" : "<animate subject> <TRANSITIVE VERB> <animate object> <verb's infinitive>", # 19
    # "via-inf" : "(Somebody ----s INFINITIVE)", # 9
    "via-inf" : "<animate subject> <INTRANSITIVE VERB> <verb's infinitive>", # 9
    # "vii-inf" : "(Something ----s INFINITIVE)", # 5
    "vii-inf" : "<inanimate subject> <INTRANSITIVE VERB> <verb's infinitive>", # 5
    # "via-for" : "(Somebody ----s for something)", # 1
    "via-for" : "<animate subject> <INTRANSITIVE VERB> for <inanimate object>", # 1
    # "via-on-anim" : "(Somebody ----s on somebody)", # 1
    "via-on-anim" : "<animate subject> <INTRANSITIVE VERB> on <animate object>", # 1
    # "via-at" : "(Somebody ----s at something)", # 1
    "via-at" : "<animate subject> <INTRANSITIVE VERB> at <inanimate object>", # 1
    # "via-out-of" : "(Somebody ----s out of somebody)", # 1
    "via-out-of" : "<animate subject> <INTRANSITIVE VERB> out of <animate object>", # 1
}


# Synset's counts by their Parts of Speech types:
# Synset's counts by their Lexical File Categories types:
dict_parts_of_speech_code_to_name = \
{
    # <Lemma writtenForm=".22-caliber" partOfSpeech="a"/>
    # <Synset id="oewn-92465616-n" ili="in" members="oewn-literary_period-n"
    #         partOfSpeech="n" lexfile="noun.time" dc:source="plWordNet 4.0">
    "n" : "noun",                # 123964 Lemmas; 84956 Synsets
    "v" : "verb",                #  11617 Lemmas; 13830 Synsets
    "a" : "adjective",           #  21624 Lemmas;  7502 Synsets
    "r" : "adverb",              #   4486 Lemmas;  3622 Synsets
    "s" : "adjective satellite", #     14 Lemmas; 10720 Synsets
}


# Sense Relation's counts by their Types:
# <SenseRelation relType="antonym" target="oewn-unequivocalness__1.07.00.."/>
# <SenseRelation relType="derivation" target="oewn-equivocal__3.00.00.."/>
# <SenseRelation relType="other" target="oewn-equal__1.18.00.." dc:type="agent"/>
# <SenseRelation relType="other" target="oewn-equivocation__1.10.00.." dc:type="event"/>
dict_relations_between_senses_name_to_descr = \
{
    "derivation" : # 74646
        "a concept which is a DERIVATIONALLY related form of a given concept",
        # https://globalwordnet.github.io/gwadoc/#derivation
        #
        # Definition:
        #
        # Derivation is a relation between two concept where Concept A is the 
        # derivationally related form of Concept B.
        #
        # Reverse: derivation
        #
        # Examples:
        #
        # yearly is the derivation of year
        # want(n) is the derivation of want(v)
        # provision is the derivation of provide
        #
    "other" : # 16556
        "any relation not otherwise specified",
        # https://globalwordnet.github.io/gwadoc/#other
        #
        # Definition:
        #
        # This is used for semantic relation types not currently supported by
        # the OMW DTD. The exact relation type can be given with dc:type.
        #
        # Reverse: other or N/A
        #
        # Examples:
        #
        # doctor other hospital
        # curator other museum
        # priest other church
        # mailman other post office
        # mayor other town hall
        # judge other court
        # ambassador other embassy
        # gardener other garden
        #
    "pertainym" : # 8072
        "a concept which is of or PERTAINING to a given concept",
        # https://globalwordnet.github.io/gwadoc/#pertainym
        #
        # Definition:
        #
        # Pertainym is a relation between two concepts where Concept A is
        # related or applicable to Concept B. Typically A will be an adjective
        # and B a noun, or A an adverb and B an adjective. It is typically used
        # for adjectives that are morphologically related to the noun they are
        # related to, are not gradable and do not have antonyms. It is also used
        # for nouns that are semantically related but not morphologically
        # related, typically because came from different languages historically,
        # so lunar for moon or arborial for tree.
        #
        # Reverse: pertainym
        #
        # Examples:
        #
        # slowly is the pertainym of slow
        # lunar is the pertainym of moon
        # naval is the pertainym of navy
        # slowly is the pertainym of slow
        # English is the pertainym of England
        # subclinical is the pertainym of clinical
        # clinical is the pertainym of clinic
        #
    "antonym" : # 7996
        "an OPPOSITE and inherently incompatible ANTONYM word",
        # https://globalwordnet.github.io/gwadoc/#antonym
        #
        # Definition:
        #
        # Two words are antonyms if their meanings are opposite in some way
        # such as:
        # * The two words show binary opposition: superior vs inferior (simple)
        # * The two words are near the opposite ends of a spectrum: hot vs cold
        #   (gradable)
        # * The two words express change or movement in opposite directions:
        #   buy vs sell (converse)
        # Antonymy can link any two members of any part-of-speech. Antonymy is
        # not transitive.
        #
        # Reverse: antonym
        #
        # Examples:
        #
        # smart has antonym stupid
        # man has antonym woman
        # superior has antonym inferior
        # buy has antonym sell
        # northen has antonym southern
        # homosexual has antonym heterosexual
        # sister has antonym brother
        #
    "exemplifies" : # 6597
        "a concept which is the example of a given GENERALIZING TYPE/CLASS concept",
        # https://globalwordnet.github.io/gwadoc/#exemplifies
        #
        # Definition:
        #
        # Exemplifies is a relation between two concepts where Concept A is
        # the example of Concept B.
        #
        # Reverse: is_exemplified_by
        #
        # Examples:
        #
        # wings exemplifies plural form
        # Band Aid exemplifies trademark
        #
    "is_exemplified_by" : # 6597
        "a concept which is the type of a given SPECIAL CASE INSTANCE concept",
        # https://globalwordnet.github.io/gwadoc/#is_exemplified_by
        #
        # Definition:
        #
        # Is Exemplified By is a relation between two concepts where Concept 
        # B is a type of Concept A, such as idiom, honorific or classifier.
        #
        # Reverse: exemplifies
        #
        # Examples:
        #
        # trademark is exemplified by Band Aid
        # plural form is exemplified by wings
        #
    "also" : # 1148
        "a word having a loose semantic relation to another WEAKLY/LOOSELY RELATED/LINKED/ASSOCIATED word",
        # https://globalwordnet.github.io/gwadoc/#also
        #
        # Definition:
        #
        # "See Also" is a self-reciprocal link (the two directions of this
        # relation share the same meaning) — Concept-A relates to Concept-B,
        # and Concept-B relates to Concept-A.
        # It denotes a relation of related meaning with another concept (going
        # beyond synonymy and similarity).
        # This link was only used to relate adjectives and verbs in Princeton
        # wordnet, but we have unconstrained this use, and we're making use of
        # this link to relate all parts-of-speech.
        #
        # Reverse: also
        #
        # Examples:
        #
        # time see also moment
        # farmer see also farmland
        # learn see also school
        # picture see also sculpture
        # plant see also flower
        # walk see also park
        #
    "participle" : # 73
        "a concept which is a PARTICIPIAL ADJECTIVE derived from a VERB expressed by a given concept",
        # https://globalwordnet.github.io/gwadoc/#participle
        #
        # Definition:
        #
        # Participle is a relation between two concepts where Concept A is a
        # participial adjective which is drived from Concept B in the form of
        # verb.
        #
        # Reverse: N/A
        #
        # Examples:
        #
        # interesting is the participial of interest
        # amazing is the participial of amaze
        #
    "similar" : # 2
        "a concept expressing CLOSELY RELATED or SOMEWHAT SIMILAR meanings, but not synonyms"
        # https://globalwordnet.github.io/gwadoc/#similar
        #
        # Definition:
        #
        # A relation between two concepts where concept A and concept B are
        # closely related in meaning but are not in the same synset. Similarity
        # is a self-reciprocal link (the two directions of this relation share
        # the same meaning) — Concept-A is similar to Concept-B, and Concept-B
        # is similar to Concept-A.
        # This link was originally used to relate adjectives, but we have
        # unconstrained this use, and we're making use of this link to relate
        # all parts-of-speech.
        # Similarity can be understood as weak synonymy, opposed to the full
        # synonymy that all lemmas in a concept must share. As adjectives are
        # not structured hierarchically (hyponymy/hypernymy) like verbs or
        # nouns, the similarity link helps showing relations between them.
        #
        # Reverse: similar
        #
        # Examples:
        #
        # tool has near_synonym instrument
        # instrument has near_synonym tools
        #
        #######################################################################
        # domain_topic
        # has_domain_topic
        # domain_region
        # has_domain_region
        # feminine
        # has_feminine
        # masculine
        # has_masculine
        # diminutive
        # has_diminutive
        # augmentative
        # has_augmentative
        # anto_gradable
        # anto_simple
        # anto_converse
}

# Sense Relation's counts by their Sub-Types:
# <SenseRelation relType="other" target="oewn-equal__1.18.00.." dc:type="agent"/>
# <SenseRelation relType="other" target="oewn-equivocation__1.10.00.." dc:type="event"/>
dict_other_relations_between_senses_name_to_descr = \
{
    "<empty>" : "", # 105131
    "event" : "a concept cna be used for defining a new EVENT/INCIDENT/EPISODE/OCCURENCE concept", # 7657
    "agent" : "a concept can be used for defining a new AGENT/ACTOR concept", # 2906
    "result" : "a concept can be used for defining a new RESULT/OUTCOME concept", # 1311
    "by_means_of" : "a concept can be used for defining a new ASSISTANCE/AID/HELP concept", # 1098
    "undergoer" : "a concept can be used for defining a new UNDERGOER/EXPERIENCING/FACING concept", # 823
    "instrument" : "a concept can be used for defining a new INSTRUMENT concept", # 783
    "uses" : "a concept can be used for defining a new USING/UTILIZING concept", # 714
    "state" : "a concept can be used for defining a new STATE concept", # 498
    "property" : "a concept can be used for defining a new PROPERTY concept", # 294
    "location" : "a concept can be used for defining a new LOCATION concept", # 260
    "vehicle" : "a concept can be used for defining a new VEHICLE concept", # 86
    "material" : "a concept can be used for defining a new MATERIAL concept", # 78
    "body_part" : "a concept can be used for defining a new BODY PART concept", # 31
    "destination" : "a concept can be used for defining a new DESTINATION concept", # 17
}

# Synset's counts by their Lexical File Categories types:
# https://wordnet.princeton.edu/documentation/lexnames5wn
# http://man.he.net/man5/lexnames
# https://stackoverflow.com/questions/42216995/
#   what-exactly-are-wordnet-lexicographer-files-understanding-how-wordnet-works
dict_synset_lexicographer_noun_files_name_to_descr = \
{
    "noun.artifact" : # 12157
        "nouns denoting man-made objects",
    "noun.person" : # 11664
        "nouns denoting people",
    "noun.plant" : # 8043
        "nouns denoting plants",
    "noun.animal" : # 7533
        "nouns denoting animals",
    "noun.act" : # 6775
        "nouns denoting acts or actions",
    "noun.communication" : # 5929
        "nouns denoting communicative processes and contents",
    "noun.state" : # 3595
        "nouns denoting stable states of affairs",
    "noun.attribute" : # 3330
        "nouns denoting attributes of people and objects",
    "noun.location" : # 3257
        "nouns denoting spatial position",
    "noun.cognition" : # 3150
        "nouns denoting cognitive processes and contents",
    "noun.substance" : # 3144
        "nouns denoting substances",
    "noun.group" : # 2718
        "nouns denoting groupings of people or objects",
    "noun.food" : # 2671
        "nouns denoting foods and drinks",
    "noun.body" : # 2037
        "nouns denoting body parts",
    "noun.object" : # 1557
        "nouns denoting natural objects (not man-made)",
    "noun.quantity" : # 1340
        "nouns denoting quantities and units of measure",
    "noun.possession" : # 1119
        "nouns denoting possession and transfer of possession",
    "noun.event" : # 1086
        "nouns denoting natural events",
    "noun.time" : # 1053
        "nouns denoting time and temporal relations",
    "noun.process" : # 799
        "nouns denoting natural processes",
    "noun.phenomenon" : # 653
        "nouns denoting natural phenomena",
    "noun.relation" : # 455
        "nouns denoting relations between people or things or ideas",
    "noun.feeling" : # 432
        "nouns denoting feelings and emotions",
    "noun.shape" : # 367
        "nouns denoting two and three dimensional shapes",
    "noun.Tops" : # 51
        "unique beginner for nouns",
    "noun.motive" : # 41
        "nouns denoting goals",
}

dict_synset_lexicographer_adj_files_name_to_descr = \
{
    # <Synset id="oewn-00001740-a" ili="i1" members="oewn-able-a"
    # partOfSpeech="a" lexfile="adj.all">
    "adj.all" : # 14499
        "all adjective clusters",
    "adj.pert" : # 3663
        "relational adjectives (pertainyms)",
    "adj.ppl" : # 60
        "participial adjectives",
}

dict_synset_lexicographer_verb_files_name_to_descr = \
{
    "verb.change" : # 2392
        "verbs of size, temperature change, intensifying, etc.",
    "verb.contact" : # 2204
        "verbs of touching, hitting, tying, digging",
    "verb.communication" : # 1565
        "verbs of telling, asking, ordering, singing",
    "verb.motion" : # 1410
        "verbs of walking, flying, swimming",
    "verb.social" : # 1112
        "verbs of political and social activities and events",
    "verb.possession" : # 848
        "verbs of buying, selling, owning",
    "verb.stative" : # 758
        "verbs of being, having, spatial relations",
    "verb.creation" : # 699
        "verbs of sewing, baking, painting, performing",
    "verb.cognition" : # 697
        "verbs of thinking, judging, analyzing, doubting",
    "verb.body" : # 550
        "verbs of grooming, dressing and bodily care",
    "verb.perception" : # 464
        "verbs of seeing, hearing, feeling",
    "verb.competition" : # 460
        "verbs of fighting, athletic activities",
    "verb.emotion" : # 345
        "verbs of feeling",
    "verb.consumption" : # 246
        "verbs of eating and drinking",
    "verb.weather" : # 80
        "verbs of raining, snowing, thawing, thundering",
}

dict_synset_lexicographer_adv_files_name_to_descr = \
{
    "adv.all" : # 3622
        "all adverbs",
}

# Synset Relation's counts by their Types:
# https://globalwordnet.github.io/gwadoc/
dict_relations_between_synsets_name_to_descr = \
{
    # <SynsetRelation relType="hyponym" target="oewn-00001930-n"/>
    "hyponym" : # 93446
        "a SUBTYPE/NARROWER/HYPONYM concept that is more SPECIFIC than a given concept",
        # https://globalwordnet.github.io/gwadoc/#hyponym
        #
        # Definition:
        #
        # A hyponym of something is its subtype: if A is a hyponym of B,
        # then all A are B.
        #
        # Reverse: hypernym
        #
        # Examples:
        #
        # dog is a hyponym of animal
        # beef is a hyponym of meat
        # pear is a hyponym of edible fruit
        # dictionary is a hyponym of wordbook
        #
    "hypernym" : # 93411
        "a SUPERTYPE/BROADER/HYPERNYM concept that is more GENERAL than a given concept",
        # "A concept with a broader meaning",
        # https://globalwordnet.github.io/gwadoc/#hypernym
        #
        # Definition:
        #
        # A hypernym of something is its supertype: if A is a hypernym of B,
        # then all B are A.
        #
        # Reverse: hyponym
        #
        # Examples:
        #
        # animal is a hypernym of dog
        # meat is a hypernym of beef
        # edible fruit is a hypernym of pear
        # wordbook is a hypernym of dictionary
        #
    "similar" : # 23129
        "a SIMILAR or CLOSELY RELATED concept, though not necessarily interchangeable",
        # https://globalwordnet.github.io/gwadoc/#similar
        #
        # Definition:
        #
        # A relation between two concepts where concept A and concept B are
        # closely related in meaning but are not in the same synset. Similarity
        # is a self-reciprocal link (the two directions of this relation share
        # the same meaning) — Concept-A is similar to Concept-B, and Concept-B
        # is similar to Concept-A.
        # This link was originally used to relate adjectives, but we have
        # unconstrained this use, and we're making use of this link to relate
        # all parts-of-speech.
        # Similarity can be understood as weak synonymy, opposed to the full
        # synonymy that all lemmas in a concept must share. As adjectives are
        # not structured hierarchically (hyponymy/hypernymy) like verbs or
        # nouns, the similarity link helps showing relations between them.
        #
        # Reverse: similar
        #
        # Examples:
        #
        # tool has near_synonym instrument
        # instrument has near_synonym tools
        #
    "holo_member" : # 12296
        "a HOMOGENIOUS BAG/SET concept that is a WHOLE COLLECTION/GROUP/HOLONYM of a given member concept",
        # "A group that this concept is a member of",
        # https://globalwordnet.github.io/gwadoc/#holo_member
        #
        # Definition:
        #
        # A relation between two concepts where concept B is a member/ element
        # of concept A. Meronym and Holonym Membership is a paired relation
        # that denotes group formation and membership. Is different from
        # hyponym as it does not relates a sub-kind of a concept. It links
        # groups to members — Concept-B is composed of many members of
        # Concept-A; and many instances of Concept-A form Concept-B.
        #
        # Reverse: mero_member
        #
        # Examples:
        #
        # player has member-holonym team
        # ship has member-holonym fleet
        #
    "mero_member" : # 12296
        "a HOMOGENIOUS ELEMENT concept that is a MEMBER/CONSTITUENT of a given collection concept",
        # "A member of this concept",
        # https://globalwordnet.github.io/gwadoc/#mero_member
        #
        # Definition:
        #
        # A relation between two concepts where concept A is a member/ element
        # of concept B. Meronym and Holonym Membership is a paired relation
        # that denotes group formation and membership. Is different from
        # hyponym as it does not relates a sub-kind of a concept. It links
        # groups to members — Many instances of Concept-A form Concept-B; and
        # Concept-B is composed of many members of Concept-A.
        #
        # Reverse: holo_member
        #
        # Examples:
        #
        # team has member-meronym player
        # fleet has member-meronym ship
        #
    "holo_part" : # 9202
        "a HETEROGENIOUS BAG/SET concept that is a WHOLE COLLECTION/GROUP/HOLONYM of a given part concept",
        # "A larger whole that this concept is part of",
        # https://globalwordnet.github.io/gwadoc/#holo_part
        #
        # Definition:
        #
        # A relation between two concepts where concept B is the whole of the
        # different component of concept A. Meronym and Holonym Part is a
        # paired relation that denotes proper parts (separable, in principle),
        # which preserve a belonging relation even if the physical link is
        # broken — Concept-A can be separated into Concept-B”; and Concept-B is
        # a part of some Concept-A.
        # This relation is also frequently used to denote geographical
        # inclusiveness relations.
        #
        # Reverse: mero_part
        #
        # Examples:
        #
        # wheel has part-holonym car
        # thumb has part-holonym glove
        #
    "mero_part" : # 9180
        "a HETEROGENIOUS COMPONENT concept that is a PART/MERONYM of a given collection concept",
        # 
        # https://globalwordnet.github.io/gwadoc/#mero_part
        #
        # Definition:
        #
        # A relation between two concepts where concept A is a component of
        # concept B. Meronym and Holonym Part is a paired relation that denotes
        # proper parts (separable, in principle), which preserve a belonging
        # relation even if the physical link is broken — Concept-A can be
        # separated into Concept-B”; and Concept-B is a part of some Concept-A.
        # This relation is also frequently used to denote geographical
        # inclusiveness relations.
        #
        # Reverse: holo_part
        #
        # Examples:
        #
        # car has part-meronym wheel
        # glove has part-meronym finger
        #
    "instance_hyponym" : # 8614
        "a INDIVIDUAL ENTITY/HYPONYM/OCCURRENCE concept that is an INSTANCE of a given type concept",
        # "An individual instance of this class",
        # https://globalwordnet.github.io/gwadoc/#instance_hyponym
        #
        # Definition:
        #
        # A relation between two concepts where concept A (instance_hyponym)
        # is a type of concept B (instance_hypernym), and where A is an
        # individual entity. A will be a terminal node in the hierarchy.
        # Instances are expressed by proper nouns.
        # Hyponymy is a relation between classes of entities. Individual
        # entities can also be said to belong to some class.
        #
        # Reverse: instance_hypernym
        #
        # Examples:
        #
        # city has instance_hyponym manchester
        #
    "instance_hypernym" : # 8614
        "a TYPE concept that is an CLASS of a given instance concept",
        # "The class of objects to which this instance belongs",
        # https://globalwordnet.github.io/gwadoc/#instance_hypernym
        #
        # Definition:
        #
        # A relation between two concepts where concept A (instance_hyponym) is
        # a type of concept B (instance_hypernym), and where A is an individual
        # entity. A will be a terminal node in the hierarchy. Instances are
        # expressed by proper nouns.
        #
        # Reverse: instance_hyponym
        #
        # Examples:
        #
        # manchester has instance_hypernym city
        #
    "has_domain_topic" : # 6905
        # "a concept which is a term in the scientific category of a given concept",
        "a SCIENTIFIC DOMAIN/SPHERE/AREA TOPIC or term concept of a given concept",
        # https://globalwordnet.github.io/gwadoc/#has_domain_topic
        #
        # Definition:
        #
        # Has Domain Topic is a relation between two concepts where Concept A
        # is a scientific category (e.g. computing, sport, biology, etc.) of
        # concept B.
        #
        # Reverse: domain_topic
        #
        # Examples:
        #
        # CPU has domain topic of computer science
        # place-kick has domain topic of football
        # evergreen has domain topic of plant
        # water has domain topic of ocean
        #
    "domain_topic" : # 6887
        "a SCIENTIFIC CATEGORY POINTER concept of a given scientific domain topic concept",
        # "a concept which is the scientific category pointer of a given concept",
        # "Indicates the category of this word",
        # https://globalwordnet.github.io/gwadoc/#domain_topic
        #
        # Definition:
        #
        # Domain Topic is a relation between two concepts where Concept B is a
        # scientific category (e.g. computing, sport, biology, etc.) of
        # concept A.
        #
        # Reverse: has_domain_topic
        #
        # Examples:
        #
        # computer science is a domain topic of CPU
        # football is a domain topic of place-kick
        # plant is a domain topic of evergreen
        # ocean is a domain topic of water
        #
    "also" : # 2728
        "a WEAKLY/SOMEWHAT/SLIGHTLY/BARELY/HARDLY RELATED concept with LOOSE/VAGUE/WEAK RELATION to a given concept",
        # "a word having a loose semantic relation to another word",
        # "See also, a reference of weak meaning",
        # https://globalwordnet.github.io/gwadoc/#also
        #
        # Definition:
        #
        # ‘See Also’ is a self-reciprocal link (the two directions of this
        # relation share the same meaning) — Concept-A relates to Concept-B,
        # and Concept-B relates to Concept-A.
        # It denotes a relation of related meaning with another concept (going
        # beyond synonymy and similarity).
        # This link was only used to relate adjectives and verbs in Princeton
        # wordnet, but we have unconstrained this use, and we're making use of
        # this link to relate all parts-of-speech.
        #
        # Reverse: also
        #
        # Examples:
        #
        # time see also moment
        # farmer see also farmland
        # learn see also school
        # picture see also sculpture
        # plant see also flower
        # walk see also park
        #
    "exemplifies" : # 1667
        "a GENERALIZING RELATION/concept to a given example concept",
        # "a concept which is the example of a given concept",
        # "Indicates the usage of this word",
        # https://globalwordnet.github.io/gwadoc/#exemplifies
        #
        # Definition:
        #
        # Exemplifies is a relation between two concepts where Concept A is the
        # example of Concept B.
        #
        # Reverse: is_exemplified_by
        #
        # Examples:
        #
        # wings exemplifies plural form
        # Band Aid exemplifies trademark
        #
    "is_exemplified_by" : # 1667
        "an EXAMPLE/SPECIALIZATION/EXEMPLIFICATION/SPECIALIZING concept which is a SPECIAL CASE for a given generalizing concept",
        # "a concept which is the type of a given concept",
        # "Indicates a word involved in the usage described by this word",
        # https://globalwordnet.github.io/gwadoc/#is_exemplified_by
        #
        # Definition:
        #
        # Is Exemplified By is a relation between two concepts where Concept B
        # is a type of Concept A, such as idiom, honorific or classifier. 
        #
        # Reverse: exemplifies
        #
        # Examples:
        #
        # trademark is exemplified by Band Aid
        # plural form is exemplified by wings
        #
    "has_domain_region" : # 1347
        #"a concept which is the term in the geographical / cultural domain of a given concept",
        "a GEOGRAPHICAL/CULTURAL DOMAIN/REGION term concept of a given concept",
        # https://globalwordnet.github.io/gwadoc/#has_domain_region
        #
        # Definition:
        #
        # Has Domain Region is a relation between two concepts where Concept A
        # is a term of the geographical / cultural domain of concept B. 
        #
        # Reverse: domain_region
        #
        # Examples:
        #
        # billion has domain region of United States
        # sushi has domain region of Japan
        # War of the Roses has domain region of England
        # Philippine Sea has domain region of Pacific
        #
    "domain_region" : # 1339
        # "a concept which is a geographical / cultural domain pointer of a given concept",
        "a GEOGRAPHICAL/CULTURAL FEATURE concept of a given geographical/cultural domain/region concept",
        # "Indicates the region of this word",
        # https://globalwordnet.github.io/gwadoc/#domain_region
        #
        # Definition:
        #
        # Domain Region is a relation between two concepts where Concept B is a
        # geographical / cultural domain of concept A. 
        #
        # Reverse: has_domain_region
        #
        # Examples:
        #
        # United States is a domain region of billion
        # Japan is a domain region of sushi
        # England is a domain region of War of the Roses
        # Pacific is a domain region of Philippine Sea
        #
    "attribute" : # 1278
        "an ATTRIBUTE/CHARACTERISTIC ABSTRACTION concept BELONGING to a given concept",
        # "A noun for which adjectives express values. The noun weight is an attribute, for which the adjectives light and heavy express values.",
        # https://globalwordnet.github.io/gwadoc/#attribute
        #
        # Definition:
        #
        # A relation between nominal and adjectival concepts where the concept
        # A is an attribute of concept B. ‘Attributes’ is a self-reciprocal
        # link (the two directions of this relation share the same meaning) —
        # Concept-A attributes to Concept-B, and Concept-B attributes to
        # Concept-A.
        # It denotes a relation between a noun and its adjectival attributes,
        # and vice-versa — for this reason it should only link adjectives to
        # nouns and vice-versa.
        #
        # Reverse: attribute
        #
        # Examples:
        #
        # fertile has attributes fecundity
        # fecundity has attributes fertile
        #
    "holo_substance" : # 830
        # "concept B is a substance of Concept A",
        "a SUM/PRODUCT/COMPOSITION concept that is a HOLONYM of a given substance concept",
        # https://globalwordnet.github.io/gwadoc/#holo_substance
        #
        # Definition:
        #
        # A relation between two concepts where concept B is made of concept A.
        # Meronym and Holonym Substance is a paired relation that denotes a
        # higher bound between part and whole. Separating/removing the
        # substance part, will change the whole — Concept-A is made of
        # Concept-B; and Concept-B is a substance of Concept-A”. 
        #
        # Reverse: mero_substance
        #
        # Examples:
        #
        # wood has substance-holonym stick
        # wood has substance-holonym beam
        #
    "mero_substance" : # 830
        # "concept A is made of concept B",
        "a SUBSTANCE concept that is a MATERIAL/MERONYM of a given product/sum/composition concept",
        # https://globalwordnet.github.io/gwadoc/#mero_substance
        #
        # Definition:
        #
        # A relation between two concepts where concept A is made of concept B.
        # Meronym and Holonym Substance is a paired relation that denotes a
        # higher bound between part and whole. Separating/removing the
        # substance part, will change the whole — Concept-A is made of
        # Concept-B; and Concept-B is a substance of Concept-A”. 
        #
        # Reverse: holo_substance
        #
        # Examples:
        #
        # stick has substance-meronym wood
        # paper has substance-meronym cellulose
        # wood has substance-meronym lignin
        #
    "is_entailed_by" : #398
        # "concept B is the result/happens because of the occurrence of A",
        "a REQUIRED/NECESSITATING concept for a given entailed concept",
        # "Opposite of entails",
        # https://globalwordnet.github.io/gwadoc/#is_entailed_by
        #
        # Definition:
        #
        # ?
        #
        # Reverse: entails
        #
        # Examples:
        #
        # sleep is entailed by snore
        #
    "entails" : # 392
        #"a verb X entails Y if X cannot be done unless Y is, or has been, done",
        "an ENTAILED concept that was entailed by a given concept",
        # "impose, involve, or imply as a necessary accompaniment or result",
        # https://globalwordnet.github.io/gwadoc/#entails
        #
        # Definition:
        #
        # Entailment is a relation that links two verbs, and it is currently
        # unilateral — Verb-A entails Verb-B, without a reciprocal or tracing
        # link. This relation presupposes/requires a semantic restriction in
        # which Verb-B has to take place before or during Verb-A.
        #
        # Reverse: is_entailed_by
        #
        # Examples:
        #
        # snore entails sleep
        #
    "is_caused_by" : # 219
        # "concept A comes about because of B",
        "an ORIGINATING/PRODUCING concept for a given caused concept",
        # https://globalwordnet.github.io/gwadoc/#is_caused_by
        #
        # Definition:
        #
        # A relation between two concepts where concept A comes into existence
        # as a result of concept B.
        #
        # Reverse: causes
        #
        # Examples:
        #
        # die is caused by kill
        #
    "causes" :  # 191
        # "concept A is an entity that produces an effect or is responsible for events or results of Concept B",
        "an IMPLIED/CAUSED concept that was caused by a given concept",
        # https://globalwordnet.github.io/gwadoc/#causes
        #
        # Definition:
        #
        # A relation between two concepts where concept B comes into existence
        # as a result of concept A. Entailment is a relation that links two
        # verbs, and it is currently unilateral — Verb-A causes Verb-B,
        # without a reciprocal or tracing link. Causation presupposes/requires
        # that some Verb-B will, inevitably, take place during or after Verb-A
        # (e.g. if Verb-A occurs, then Verb-B will also occur).
        # While not exclusive to these types of verbs, many verbs that have
        # both a transitive and an intransitive form will frequently be
        # submitted to this relation.
        #
        # Reverse: is_caused_by
        #
        # Examples:
        #
        # kill causes die
        #
    ###########################################################################
    # eq_synonym (reverse: eq_synonym):
        # "A and B are equivalent concepts but their nature requires that they
        #  remain separate (e.g. Exemplifies)"
        # people has equal-synonym folks
        # cop has equal-synonym policeman
        # fiddle has equal-synonym violin
        # begin has equal-synonym start
    # meronym
    # holonym
    # mero_location
    # holo_location
    # mero_portion
    # holo_portion
    # mero_substance
    # holo_substance
    # other
    # also
    # state_of
    # be_in_state
    # subevent
    # is_subevent_of
    # manner_of
    # in_manner
    # attribute
    # restricts
    # restricted_by
    # classifies
    # classified_by
    # entails
    # is_entailed_by
    # domain
    # has_domain
    # domain_region
    # has_domain_region
    # role
    # involved
    # agent
    # involved_agent
    # patient
    # involved_patient
    # result
    # involved_result
    # instrument
    # involved_instrument
    # location
    # involved_location
    # direction
    # involved_direction
    # target_direction
    # involved_target_direction
    # source_direction
    # involved_source_direction
    # co_role
    # co_agent_patient
    # co_patient_agent
    # co_agent_instrument
    # co_instrument_agent
    # co_agent_result
    # co_result_agent
    # co_patient_instrument
    # co_instrument_patient
    # co_result_instrument
    # co_instrument_result
    # feminine
    # has_feminine
    # masculine
    # has_masculine
    # young
    # has_young
    # anto_gradable
    # anto_simple
    # anto_converse (reverse: anto_converse)
        # "word pairs that name or describe a single relationship from opposite
        #  perspectives"
        # parent is a converse antonym of child
    # ir_synonym (reverse: ir_synonym)
        # "A concept that means the same except for the style or connotation"
        # loot is an inter-register synonym of money
}


###############################################################################


class Lexicon :

    def __init__(self, str_identifier) :
        self._str_identifier = str_identifier
        self._dict_lexical_entries = {}
        self._dict_senses = {}
        self._dict_synsets = {}
        self._dict_syntactic_behaviours = {}
        # Summary counts collections:
        self._dict_syntactic_behaviour_types_counts = {} # subtype
        self._dict_sense_relation_types_counts = {} # relType
        self._dict_sense_relation_subtypes_counts = {} # dc:type
        self._dict_lemma_parts_of_speech_counts = {} # partOfSpeech
        self._dict_synset_parts_of_speech_counts = {} # partOfSpeech
        self._dict_synset_lexical_file_categories_counts = {} # lexfuile
        self._dict_synset_relation_types_counts = {} # relType

    def add_lexical_entry(self, lexical_entry_obj,) :
        self._dict_lexical_entries[
            lexical_entry_obj.identifier] = lexical_entry_obj

    def add_sense(self, sense_obj,) :
        self._dict_senses[sense_obj.identifier] = sense_obj

    def add_synset(self, synset_obj,) :
        self._dict_synsets[synset_obj.identifier] = synset_obj

    def add_syntactic_behaviour(self, syntactic_behaviour_obj,) :
        self._dict_syntactic_behaviours[
            syntactic_behaviour_obj.identifier] = syntactic_behaviour_obj

    @property
    def identifier(self,) :
        return self._str_identifier

    @property
    def dictionary_of_lexical_entries(self) :
        return self._dict_lexical_entries

    @property
    def dictionary_of_senses(self) :
        return self._dict_senses

    @property
    def dictionary_of_synsets(self) :
        return self._dict_synsets

    @property
    def dictionary_of_syntactic_behaviours(self) :
        return self._dict_syntactic_behaviours

    def refresh_summary_counts(self,) :
        # bool_debug_traces = False
        self._dict_lemma_parts_of_speech_counts = {} # partOfSpeech
        self._dict_syntactic_behaviour_types_counts = {} # subtype
        self._dict_sense_relation_types_counts = {} # relType
        self._dict_sense_relation_subtypes_counts = {} # dc:type
        for lexical_entry_obj in self._dict_lexical_entries.values() :
            str_part_of_speech = lexical_entry_obj.part_of_speech
            if str_part_of_speech in self._dict_lemma_parts_of_speech_counts :
                self._dict_lemma_parts_of_speech_counts[str_part_of_speech] += 1
            else :
                self._dict_lemma_parts_of_speech_counts[str_part_of_speech] = 1
            for sense_obj in lexical_entry_obj.dictionary_of_senses.values() :
                #if bool_debug_traces :
                #    print("Source Sense: " + sense_obj.identifier)
                for str_syntactic_behaviour_identifier in sense_obj.list_of_syntactic_behaviour_identifiers :
                    str_syntactic_behaviour_info = str_syntactic_behaviour_identifier + " (" + \
                        self._dict_syntactic_behaviours[
                            str_syntactic_behaviour_identifier].subcategorization_frame + ")"
                    if str_syntactic_behaviour_info in self._dict_syntactic_behaviour_types_counts :
                        self._dict_syntactic_behaviour_types_counts[str_syntactic_behaviour_info] += 1
                    else :
                        self._dict_syntactic_behaviour_types_counts[str_syntactic_behaviour_info] = 1
                for sense_relation_obj in sense_obj.dictionary_of_sense_relations.values() :
                    str_sense_relation_type = sense_relation_obj.sense_relation_type
                    #if bool_debug_traces :
                    #    print("Sense Relation Type: " + str_sense_relation_type)
                    if str_sense_relation_type in self._dict_sense_relation_types_counts :
                        self._dict_sense_relation_types_counts[str_sense_relation_type] += 1
                    else :
                        self._dict_sense_relation_types_counts[str_sense_relation_type] = 1
                    str_sense_relation_subtype = sense_relation_obj.sense_relation_subtype
                    #if bool_debug_traces :
                    #    print("Sense Relation Sub-Type: " + str_sense_relation_subtype)
                    if str_sense_relation_subtype in self._dict_sense_relation_subtypes_counts :
                        self._dict_sense_relation_subtypes_counts[str_sense_relation_subtype] += 1
                    else :
                        self._dict_sense_relation_subtypes_counts[str_sense_relation_subtype] = 1
        self._dict_syntactic_behaviour_types_counts = dict(sorted(
            self._dict_syntactic_behaviour_types_counts.items(),
            key = lambda item : item[1], reverse = True))
        self._dict_sense_relation_types_counts = dict(sorted(
            self._dict_sense_relation_types_counts.items(),
            key = lambda item : item[1], reverse = True))
        self._dict_sense_relation_subtypes_counts = dict(sorted(
            self._dict_sense_relation_subtypes_counts.items(),
            key = lambda item : item[1], reverse = True))
        self._dict_lemma_parts_of_speech_counts = dict(sorted(
            self._dict_lemma_parts_of_speech_counts.items(),
            key = lambda item : item[1], reverse = True))
        #
        self._dict_synset_parts_of_speech_counts = {} # partOfSpeech
        self._dict_synset_lexical_file_categories_counts = {} # lexfile
        self._dict_synset_relation_types_counts = {} # relType
        for synset_obj in self._dict_synsets.values() :
            str_part_of_speech = synset_obj.part_of_speech
            if str_part_of_speech in self._dict_synset_parts_of_speech_counts :
                self._dict_synset_parts_of_speech_counts[str_part_of_speech] += 1
            else :
                self._dict_synset_parts_of_speech_counts[str_part_of_speech] = 1
            str_lexical_file_category = synset_obj.lexical_file_category
            if str_lexical_file_category in self._dict_synset_lexical_file_categories_counts :
                self._dict_synset_lexical_file_categories_counts[str_lexical_file_category] += 1
            else :
                self._dict_synset_lexical_file_categories_counts[str_lexical_file_category] = 1
            for synset_relation_obj in synset_obj.dictionary_of_synset_relations.values() :
                str_synset_relation_type = synset_relation_obj.synset_relation_type
                if str_synset_relation_type in self._dict_synset_relation_types_counts :
                    self._dict_synset_relation_types_counts[str_synset_relation_type] += 1
                else :
                    self._dict_synset_relation_types_counts[str_synset_relation_type] = 1
        self._dict_synset_parts_of_speech_counts = dict(sorted(
            self._dict_synset_parts_of_speech_counts.items(),
            key = lambda item : item[1], reverse = True))
        self._dict_synset_lexical_file_categories_counts = dict(sorted(
            self._dict_synset_lexical_file_categories_counts.items(),
            key = lambda item : item[1], reverse = True))
        self._dict_synset_relation_types_counts = dict(sorted(
            self._dict_synset_relation_types_counts.items(),
            key = lambda item : item[1], reverse = True))

    def print_summary(self,) :
        print('Lexicon "{:s}" counts by its element types:'.format(
            self.identifier))
        print()
        print('LexicalEntry: {:d}'.format(len(self._dict_lexical_entries)))
        print('Synset: {:d}'.format(len(self._dict_synsets)))
        print('SyntacticBehaviour: {:d}'.format(len(self._dict_syntactic_behaviours)))
        print()
        print("Lemma's counts by their Parts of Speech types:")
        print()
        for (str_part_of_speech_name, str_int_lemma_count) in \
           self._dict_lemma_parts_of_speech_counts.items() :
            print("{:s}: {:d}".format(str_part_of_speech_name, str_int_lemma_count))
        print()
        print("Synset's counts by their Parts of Speech types:")
        print()
        for (str_part_of_speech_name, str_int_synset_count) in \
           self._dict_synset_parts_of_speech_counts.items() :
            print("{:s}: {:d}".format(str_part_of_speech_name, str_int_synset_count))
        print()
        print("Sense Relations counts by their Types:")
        print()
        for (str_sense_relation_type, str_int_sense_relation_count) in \
           self._dict_sense_relation_types_counts.items() :
            print("{:s}: {:d}".format(str_sense_relation_type, str_int_sense_relation_count))
        print()
        print('Sense Relations counts by their Sub-Types for Type "Other":')
        print()
        for (str_sense_relation_subtype, str_int_sense_relation_count) in \
           self._dict_sense_relation_subtypes_counts.items() :
            print("{:s}: {:d}".format(str_sense_relation_subtype, str_int_sense_relation_count))
        print()
        print("Synset Relation's counts by their Types:")
        print()
        for (str_synset_relation_type, str_int_synset_count) in \
           self._dict_synset_relation_types_counts.items() :
            print("{:s}: {:d}".format(str_synset_relation_type, str_int_synset_count))
        print()
        print("Synset's counts by their Lexical File Categories types:")
        print()
        for (str_synset_lexical_file_category, str_int_synset_count) in \
           self._dict_synset_lexical_file_categories_counts.items() :
            print("{:s}: {:d}".format(str_synset_lexical_file_category, str_int_synset_count))
        print()
        print("Sense Relation's counts by their Syntactic Behaviour types:")
        print()
        for (str_syntactic_behaviour_type, str_int_syntactic_behaviour_count) in \
           self._dict_syntactic_behaviour_types_counts.items() :
            print("{:s}: {:d}".format(str_syntactic_behaviour_type, str_int_syntactic_behaviour_count))


    def save_to_pickle(self, str_file_name) :
        with open(str_file_name + ".pickle", 'wb') as fp :
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(str_file_name) :
        with open(str_file_name + ".pickle", 'rb') as fp:
            return pickle.load(fp)

    def save_to_lzma(self, str_file_path, str_file_name) :
        # https://stackoverflow.com/questions/57983431/
        # whats-the-most-space-efficient-way-to-compress-serialized-python-data
        with lzma.open(os.path.join(str_file_path, str_file_name + ".xz"), 'wb') as fp :
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    def load_from_lzma(str_file_path, str_file_name) :
        with lzma.open(os.path.join(str_file_path, str_file_name + ".xz"), 'rb') as fp:
            return pickle.load(fp)

    def load_from_xml(str_file_path, str_file_name) :

        tree = ET.parse(os.path.join(str_file_path, str_file_name + ".xml"))
        xml_elem_LexicalResource = tree.getroot()

        # tag, attrib, text
        if xml_elem_LexicalResource.tag != "LexicalResource" :
            raise TypeError(
                'Error: expecting "LexicalResource" type tag.')
        # <LexicalResource xmlns:dc="https://globalwordnet.github.io/schemas/dc/">
        str_dc_type = "{https://globalwordnet.github.io/schemas/dc/}type" # "dc:type"
        xml_elem_Lexicon = xml_elem_LexicalResource[0]
        if xml_elem_Lexicon.tag != "Lexicon" :
            raise TypeError('Error: expecting "Lexicon" type tag.')
        if not "id" in xml_elem_Lexicon.attrib :
            raise TypeError(
                'Error: expecting "id" attribute in "Lexicon" tag.')
        lexicon = Lexicon(str_identifier = xml_elem_Lexicon.attrib["id"])

        bool_Lexicon_has_LexicalEntry = False
        for xml_elem_Lexicon_child in xml_elem_Lexicon :
            str_Lexicon_child_tag = xml_elem_Lexicon_child.tag
            if str_Lexicon_child_tag == "LexicalEntry" :
                xml_elem_LexicalEntry = xml_elem_Lexicon_child
                if not "id" in xml_elem_LexicalEntry.attrib :
                    raise TypeError(
                        'Error: expecting "id" attribute in "LexicalEntry" tag.')
                str_Lemma_written_form = None
                str_Lemma_part_of_speech = None
                bool_LexicalEntry_has_one_Lemma = False
                lst_Forms = []
                lst_Senses = []
                lst_Pronunciations = []
                for xml_elem_LexicalEntry_child in xml_elem_LexicalEntry :
                    str_LexicalEntry_child_tag = xml_elem_LexicalEntry_child.tag
                    if str_LexicalEntry_child_tag == "Lemma" :
                        xml_elem_Lemma = xml_elem_LexicalEntry_child
                        if bool_LexicalEntry_has_one_Lemma :
                            raise ValueError(
                                'Error: expecting only one "Lemma" tag in "LexicalEntry".')
                        if not "writtenForm" in xml_elem_Lemma.attrib :
                            raise TypeError(
                                'Error: expecting "writtenForm" attribute in "Lemma" tag.')
                        if not "partOfSpeech" in xml_elem_Lemma.attrib :
                            raise TypeError(
                                'Error: expecting "partOfSpeech" attribute in "Lemma" tag.')
                        str_Lemma_written_form = xml_elem_Lemma.attrib["writtenForm"]
                        str_Lemma_part_of_speech = xml_elem_Lemma.attrib["partOfSpeech"]
                        bool_LexicalEntry_has_one_Lemma = True
                        for xml_elem_Lemma_child in xml_elem_Lemma :
                            str_Lemma_child_tag = xml_elem_Lemma_child.tag
                            if str_Lemma_child_tag == "Pronunciation" :
                                xml_elem_Pronunciation = xml_elem_Lemma_child
                                if "variety" in xml_elem_Pronunciation.attrib :
                                    str_variety = xml_elem_Pronunciation.attrib["variety"]
                                else :
                                    str_variety = None
                                pronunciation_obj = Pronunciation(
                                    str_text = xml_elem_Pronunciation.text,
                                    str_variety = str_variety,)
                                lst_Pronunciations.append(pronunciation_obj)
                    elif str_LexicalEntry_child_tag == "Form" :
                        xml_elem_Form = xml_elem_LexicalEntry_child
                        if not "writtenForm" in xml_elem_Form.attrib :
                            raise TypeError(
                                'Error: expecting "writtenForm" attribute in "Form" tag.')
                        str_Form_written_form = xml_elem_Form.attrib["writtenForm"]
                        form_obj = Form(str_written_form = str_Form_written_form)
                        lst_Forms.append(form_obj)
                    elif str_LexicalEntry_child_tag == "Sense" :
                        xml_elem_Sense = xml_elem_LexicalEntry_child
                        if not "id" in xml_elem_Sense.attrib :
                            raise TypeError(
                                'Error: expecting "id" attribute in "Sense" tag.')
                        if not "synset" in xml_elem_Sense.attrib :
                            raise TypeError(
                                'Error: expecting "synset" attribute in "Sense" tag.')
                        sense_obj = Sense(
                            str_identifier = xml_elem_Sense.attrib["id"],
                            str_synset_identifier = xml_elem_Sense.attrib["synset"],)
                        for xml_elem_Sense_child in xml_elem_Sense :
                            str_Sense_child_tag = xml_elem_Sense_child.tag
                            if str_Sense_child_tag == "SenseRelation" :
                                xml_elem_SenseRelation = xml_elem_Sense_child
                                if not "target" in xml_elem_SenseRelation.attrib :
                                    raise TypeError(
                                        'Error: expecting "target" attribute in "SenseRelation" tag.')
                                if not "relType" in xml_elem_SenseRelation.attrib :
                                    raise TypeError(
                                        'Error: expecting "relType" attribute in "SenseRelation" tag.')
                                if str_dc_type in xml_elem_SenseRelation.attrib :
                                    str_sense_relation_subtype = xml_elem_SenseRelation.attrib[str_dc_type]
                                else :
                                    str_sense_relation_subtype = "<empty>"
                                sense_relation_obj = SenseRelation(
                                    str_source_sense_identifier = sense_obj.identifier,
                                    str_target_sense_identifier = xml_elem_SenseRelation.attrib["target"],
                                    str_sense_relation_type = xml_elem_SenseRelation.attrib["relType"],
                                    str_sense_relation_subtype = str_sense_relation_subtype)
                                sense_obj.add_sense_relation(sense_relation_obj)
                        if "subcat" in xml_elem_Sense.attrib :
                            lst_str_syntactic_behaviour_identifiers = \
                                xml_elem_Sense.attrib["subcat"].split()
                            for str_syntactic_behaviour_identifier in lst_str_syntactic_behaviour_identifiers :
                                sense_obj.add_syntactic_behaviour(
                                    str_syntactic_behaviour_identifier = str_syntactic_behaviour_identifier)
                        lst_Senses.append(sense_obj)
                if not bool_LexicalEntry_has_one_Lemma :
                    raise ValueError(
                        'Error: expecting one and only one "Lemma" tag in "LexicalEntry".')
                lexical_entry_obj = LexicalEntry(
                    str_identifier = xml_elem_LexicalEntry.attrib["id"],
                    str_written_form = str_Lemma_written_form,
                    str_part_of_speech = str_Lemma_part_of_speech)
                for pronunciation_obj in lst_Pronunciations :
                    lexical_entry_obj.lemma.add_pronunciation(
                        pronunciation_obj = pronunciation_obj)
                for form_obj in lst_Forms :
                    lexical_entry_obj.add_form(form_obj = form_obj)
                for sense_obj in lst_Senses :
                    lexical_entry_obj.add_sense(sense_obj = sense_obj)
                    lexicon.add_sense(sense_obj = sense_obj)
                lexicon.add_lexical_entry(lexical_entry_obj = lexical_entry_obj)
                bool_Lexicon_has_LexicalEntry = True
            elif str_Lexicon_child_tag == "Synset" :
                xml_elem_Synset = xml_elem_Lexicon_child
                if not "id" in xml_elem_Synset.attrib :
                    raise TypeError(
                        'Error: expecting "id" attribute in "Synset" tag.')
                if not "partOfSpeech" in xml_elem_Synset.attrib :
                    raise TypeError(
                        'Error: expecting "partOfSpeech" attribute in "Synset" tag.')
                if not "lexfile" in xml_elem_Synset.attrib :
                    raise TypeError(
                        'Error: expecting "lexfile" attribute in "Synset" tag.')
                if not "members" in xml_elem_Synset.attrib :
                    raise TypeError(
                        'Error: expecting "members" attribute in "Synset" tag.')
                synset_obj = Synset(
                    str_identifier = xml_elem_Synset.attrib["id"],
                    str_part_of_speech = xml_elem_Synset.attrib["partOfSpeech"],
                    str_lexical_file_category = xml_elem_Synset.attrib["lexfile"])
                lst_str_lexical_entries_identifiers = \
                    xml_elem_Synset.attrib["members"].split()
                for str_lexical_entry_identifier in lst_str_lexical_entries_identifiers :
                    synset_obj.add_lexical_entry_identifier(
                        str_lexical_entry_identifier = str_lexical_entry_identifier)
                for xml_elem_Synset_child in xml_elem_Synset :
                    str_Synset_child_tag = xml_elem_Synset_child.tag
                    if str_Synset_child_tag == "Definition" :
                        xml_elem_Definition = xml_elem_Synset_child
                        synset_obj.add_definition(str_definition = Definition(
                            str_text = xml_elem_Definition.text))
                    elif str_Synset_child_tag == "Example" :
                        xml_elem_Example = xml_elem_Synset_child
                        synset_obj.add_example(str_example = Example(
                            str_text = xml_elem_Example.text))
                    elif str_Synset_child_tag == "SynsetRelation" :
                        xml_elem_SynsetRelation = xml_elem_Synset_child
                        if not "relType" in xml_elem_SynsetRelation.attrib :
                            raise TypeError(
                                'Error: expecting "relType" attribute in "SynsetRelation" tag.')
                        if not "target" in xml_elem_SynsetRelation.attrib :
                            raise TypeError(
                                'Error: expecting "target" attribute in "SynsetRelation" tag.')
                        synset_relation_obj = SynsetRelation(
                            str_source_synset_identifier = synset_obj.identifier,
                            str_target_synset_identifier = xml_elem_SynsetRelation.attrib["target"],
                            str_synset_relation_type = xml_elem_SynsetRelation.attrib["relType"])
                        synset_obj.add_synset_relation(
                            synset_relation_obj = synset_relation_obj)
                lexicon.add_synset(synset_obj = synset_obj)
            elif str_Lexicon_child_tag == "SyntacticBehaviour" :
                xml_elem_SyntacticBehaviour = xml_elem_Lexicon_child
                if not "id" in xml_elem_SyntacticBehaviour.attrib :
                    raise TypeError(
                        'Error: expecting "id" attribute in "SyntacticBehaviour" tag.')
                if not "subcategorizationFrame" in xml_elem_SyntacticBehaviour.attrib :
                    raise TypeError(
                        'Error: expecting "subcategorizationFrame" attribute in ' +
                        '"SyntacticBehaviour" tag.')
                syntactic_behaviour_obj = SyntacticBehaviour(
                    str_identifier = xml_elem_SyntacticBehaviour.attrib["id"],
                    str_subcategorization_frame = xml_elem_SyntacticBehaviour.attrib[
                        'subcategorizationFrame'])
                lexicon.add_syntactic_behaviour(
                    syntactic_behaviour_obj = syntactic_behaviour_obj)

            if not bool_Lexicon_has_LexicalEntry :
                raise TypeError(
                    'Error: expecting at least one "LexicalEntry" tag in "Lexicon" tag.')
        lexicon.refresh_summary_counts()
        return lexicon


class LexicalEntry :

    def __init__(self, str_identifier, str_written_form, str_part_of_speech) :
        self._str_identifier = str_identifier
        self._lemma = Lemma(str_written_form, str_part_of_speech)
        self._dict_senses = {}
        self._list_forms = []

    def add_sense(self, sense_obj,) :
        self._dict_senses[sense_obj.identifier] = sense_obj

    def add_form(self, form_obj,) :
        self._list_forms.append(form_obj)

    @property
    def identifier(self,) :
        return self._str_identifier

    @property
    def lemma(self,) :
        return self._lemma

    @property
    def written_form(self,) :
        return self._lemma.written_form

    @property
    def part_of_speech(self,) :
        return self._lemma.part_of_speech

    @property
    def dictionary_of_senses(self,) :
        return self._dict_senses

    @property
    def list_of_forms(self,) :
        return self._list_forms


class Lemma :

    def __init__(self, str_written_form, str_part_of_speech) :
        self._str_written_form = str_written_form
        self._str_part_of_speech = str_part_of_speech
        self._list_pronunciations = []

    def add_pronunciation(self, pronunciation_obj,) :
        self._list_pronunciations.append(pronunciation_obj)

    @property
    def written_form(self,) :
        return self._str_written_form

    @property
    def part_of_speech(self,) :
        return self._str_part_of_speech

    @property
    def list_of_pronunciations(self,) :
        return self._list_pronunciations


class Pronunciation :

    def __init__(self, str_text, str_variety = None,) :
        self._str_text = str_text
        self._str_variety = str_variety

    @property
    def text(self) :
        return self._str_text

    @property
    def variety(self) :
        return self._str_variety


class Form :

    def __init__(self, str_written_form) :
        self._str_written_form = str_written_form

    @property
    def written_form(self,) :
        return self._str_written_form


class Sense :

    def __init__(self, str_identifier, str_synset_identifier) :
        self._str_identifier = str_identifier
        self._str_synset_identifier = str_synset_identifier
        self._dict_sense_relations = {}
        self._list_syntactic_behaviour_identifiers = [] # subcat

    def add_sense_relation(self, sense_relation_obj,) :
        self._dict_sense_relations[
            sense_relation_obj.target_sense_identifier + "|" +
            sense_relation_obj.sense_relation_type] = sense_relation_obj

    def add_syntactic_behaviour(self, str_syntactic_behaviour_identifier,) :
        self._list_syntactic_behaviour_identifiers.append(
            str_syntactic_behaviour_identifier)

    @property
    def identifier(self,) :
        return self._str_identifier

    @property
    def synset_identifier(self,) :
        return self._str_synset_identifier

    @property
    def dictionary_of_sense_relations(self,) :
        return self._dict_sense_relations

    @property
    def list_of_syntactic_behaviour_identifiers(self,) :
        return self._list_syntactic_behaviour_identifiers


class SyntacticBehaviour :

    def __init__(self, str_identifier, str_subcategorization_frame) :
        self._str_identifier = str_identifier
        self._str_subcategorization_frame = str_subcategorization_frame

    @property
    def identifier(self,) :
        return self._str_identifier

    @property
    def subcategorization_frame(self,) :
        return self._str_subcategorization_frame


class SenseRelation :

    def __init__(
            self, str_source_sense_identifier, str_target_sense_identifier,
            str_sense_relation_type, str_sense_relation_subtype) :
        self._str_source_sense_identifier = str_source_sense_identifier
        self._str_target_sense_identifier = str_target_sense_identifier
        self._str_sense_relation_type = str_sense_relation_type # relType
        self._str_sense_relation_subtype = str_sense_relation_subtype # dc:type

    @property
    def source_sense_identifier(self,) :
        return self._str_source_sense_identifier

    @property
    def target_sense_identifier(self,) :
        return self._str_target_sense_identifier

    @property
    def sense_relation_type(self,) :
        return self._str_sense_relation_type

    @property
    def sense_relation_subtype(self,) :
        return self._str_sense_relation_subtype


class Synset :

    def __init__(
            self, str_identifier, str_part_of_speech,
            str_lexical_file_category) :
        self._str_identifier = str_identifier
        self._str_part_of_speech = str_part_of_speech
        self._str_lexical_file_category = str_lexical_file_category
        self._list_lexical_entries_identifiers = []
        self._list_definitions = []
        self._list_examples = []
        self._dict_synset_relations = {}

    def add_lexical_entry_identifier(self, str_lexical_entry_identifier,) :
        self._list_lexical_entries_identifiers.append(
            str_lexical_entry_identifier)

    def add_definition(self, str_definition,) :
        self._list_definitions.append(str_definition)

    def add_example(self, str_example,) :
        self._list_examples.append(str_example)

    def add_synset_relation(self, synset_relation_obj,) :
        self._dict_synset_relations[
            synset_relation_obj.target_synset_identifier + "|" +
            synset_relation_obj.synset_relation_type] = synset_relation_obj

    @property
    def identifier(self) :
        return self._str_identifier

    @property
    def part_of_speech(self) :
        return self._str_part_of_speech

    @property
    def lexical_file_category(self) :
        return self._str_lexical_file_category

    @property
    def list_of_lexical_entries_identifiers(self) :
        return self._list_lexical_entries_identifiers

    @property
    def list_of_definitions(self) :
        return self._list_definitions

    @property
    def list_of_examples(self) :
        return self._list_examples

    @property
    def dictionary_of_synset_relations(self) :
        return self._dict_synset_relations


class Definition :

    def __init__(self, str_text) :
        self._str_text = str_text

    @property
    def text(self) :
        return self._str_text


class Example :
    def __init__(self, str_text) :
        self._str_text = str_text

    @property
    def text(self) :
        return self._str_text


class SynsetRelation :

    def __init__(
            self, str_source_synset_identifier, str_target_synset_identifier,
            str_synset_relation_type) :
        self._str_source_synset_identifier = str_source_synset_identifier
        self._str_target_synset_identifier = str_target_synset_identifier
        self._str_synset_relation_type = str_synset_relation_type # relType

    @property
    def source_synset_identifier(self,) :
        return self._str_source_synset_identifier

    @property
    def target_synset_identifier(self,) :
        return self._str_target_synset_identifier

    @property
    def synset_relation_type(self,) :
        return self._str_synset_relation_type


###############################################################################


def main() -> int :

    lexicon = Lexicon.load_from_xml(
        str_file_path = "data_xml_wordnet",
        str_file_name = "wn")
    lexicon.print_summary()

    print()
    print(type(lexicon))

    lexicon.save_to_lzma(
        str_file_path = DIR_PATH_DATA_PKL_XZ_WORDNET,
        str_file_name = FILE_NAME_DATA_PKL_XZ_WORDNET)
    del lexicon
    lexicon1 = Lexicon.load_from_lzma(
        str_file_path = DIR_PATH_DATA_PKL_XZ_WORDNET,
        str_file_name = FILE_NAME_DATA_PKL_XZ_WORDNET)

    print(type(lexicon1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
