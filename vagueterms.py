import nltk
import language_check
import os
import glob
import string
from nltk.corpus import wordnet as wn
from textstat import textstat
import csv
import re
from spellchecker import SpellChecker


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def getSize(text):
    words = text.split()
    return len(words)

def tokenizeSentence(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenized_sent = sent_detector.tokenize(text.strip())

    return tokenized_sent


def getComplexity(text):
    return textstat.dale_chall_readability_score(text)


def getAmbiguity(text):
    words = text.split()
    word_count = len(words)

    tokenized_sent = tokenizeSentence(text)

    ambwordlist = set()
    treebank_pos_tags = []

    for sent in tokenized_sent:
        words = nltk.word_tokenize(sent)
        treebank_pos_tags.append(nltk.pos_tag(words))

        for pos_tag in treebank_pos_tags:
            for word, pos in pos_tag:
                wn_pos_tag = get_wordnet_pos(pos)

                if wn_pos_tag != '':
                    syns = wn.synsets(str(word), pos=wn_pos_tag)

                    if len(syns) > 1:
                        ambwordlist.add(str(word))

    freq = len(ambwordlist) / word_count
    return freq

def getPunctuationFrequency(text):
    tokenized_sent = tokenizeSentence(text)
    length = len(text)
    punctuation_count = 0

    for sent in tokenized_sent:
        for i in sent:
            if i in string.punctuation and i != '.':
                punctuation_count += 1

    freq = punctuation_count / length
    return freq

def getAcronyms(text):
    pattern = r'(?:(?<=\.|\s)[A-Z]\.)+'

    tokenized_sent = tokenizeSentence(text)
    words = text.split()
    length = len(words)

    acronyms = []

    for sent in tokenized_sent:
        acronyms.append(re.findall(pattern, sent))

    freq = len(acronyms) / length
    return freq


#IMPRECISE WORDS
modalWords = ['may', 'might', 'can', 'could', 'would', 'likely']
conditionWords = ['depending', 'necessary', 'appropriate', 'inappropriate', 'as needed',
                  'as applicable', 'otherwise reasonably', 'sometimes', 'from time to time']
generalizationWords = ['generally', 'mostly', 'widely', 'general', 'commonly',
                       'usually', 'normally', 'typically', 'largely', 'often', 'primarily', 'among other things']
numericWords = ['anyone', 'certain', 'everyone', 'numerous', 'some',
                'most', 'few', 'much', 'many', 'various', 'including but not limited to']
probableWords = ['probably', 'possibly', 'optionally']
usableWords = ['adaptable', 'extensible', 'easy', 'familiar']

#CONNECTIVE WORDS
copulativeWords = ['and', 'both', 'as well as', 'not only', 'but also']
controlflowWords = ['if', 'then', 'while']
anaphoricalWords = ['it', 'this', 'those']


def getConnectiveWords(text):
    connective_word_count = 0

    words = text.split()
    length = len(words)

    for word in copulativeWords:
        answer = text.count(word)
        connective_word_count += answer
    for word in controlflowWords:
        answer = text.count(word)
        connective_word_count += answer
    for word in anaphoricalWords:
        answer = text.count(word)
        connective_word_count += answer

    freq = connective_word_count / length
    return freq


def getImpreciseWords(text):
    imprecise_word_count = 0

    words = text.split()
    length = len(words)

    for word in modalWords:
        answer = text.count(word)
        imprecise_word_count += answer
    for word in conditionWords:
        answer = text.count(word)
        imprecise_word_count += answer
    for word in generalizationWords:
        answer = text.count(word)
        imprecise_word_count += answer
    for word in numericWords:
        answer = text.count(word)
        imprecise_word_count += answer
    for word in probableWords:
        answer = text.count(word)
        imprecise_word_count += answer
    for word in usableWords:
        answer = text.count(word)
        imprecise_word_count += answer

    freq = imprecise_word_count / length
    return freq


def checkgrammar(text):
    tool = language_check.LanguageTool('en-US')

    tokenized_sent = tokenizeSentence(text)
    length = len(tokenized_sent)
    i = 0

    for sent in tokenized_sent:
        matches = tool.check(sent)
        i = i + len(matches)

    freq = i/length
    return freq

def spellChecker(text):
    spell = SpellChecker()

    words = text.split()
    misspelled = spell.unknown(words)
    length = len(words)

    freq = len(misspelled)/length
    return freq



data_pp = 'data_pp.csv'
data_tos = 'data_tos.csv'

pp_path = '/Users/anantaa/Desktop/python/readability/privacy_policy'
tos_path = '/Users/anantaa/Desktop/python/readability/ToS'


def getFileName(fileName):
    fileName = re.sub(tos_path, "", fileName)
    fileName = fileName.strip(".txt")
    return fileName.strip("/")


def main():
    filename = []
    size = []
    complexity = []
    ambiguity = []
    punc_freq = []
    acronym = []
    connective_wrd = []
    imprecise_wrd = []
    grammar_chk = []
    spell_chk = []

    for file in glob.glob(os.path.join(tos_path, '*.txt')):
        f = open(file, 'r', encoding='latin-1')
        text = f.read()
        f.close()

        filename.append(getFileName(file))
        size.append(getSize(text))
        complexity.append(getComplexity(text))
        ambiguity.append(getAmbiguity(text))
        punc_freq.append(getPunctuationFrequency(text))
        acronym.append(getAcronyms(text))
        connective_wrd.append(getConnectiveWords(text))
        imprecise_wrd.append(getImpreciseWords(text))
        grammar_chk.append(checkgrammar(text))
        spell_chk.append(spellChecker(text))

        print(getFileName(file))

    with open(data_tos, mode='w') as data_file:
        wr = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(filename)):
            wr.writerow([filename[i], size[i], complexity[i], ambiguity[i], punc_freq[i], acronym[i], connective_wrd[i],
                         imprecise_wrd[i], grammar_chk[i], spell_chk[i]])


main()

