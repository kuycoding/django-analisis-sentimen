import re
import string

import pandas as pd
import swifter
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from django import template
from django.template.defaultfilters import stringfilter
nltk.download()

register = template.Library()

@register.simple_tag
def clean(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:/\/\S+)", " ", str(text)).split())

# remove number
def remove_number(text):
    return re.sub(r"\d+", "", text)

# remove punctuation (emoji)
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# remove whitespace depan & belakang
def remove_whitespace_LT(text):
    return text.strip()

# remove multiple whitespace menjadi satu spasi
def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)

def casefolding(text):
    tweet = text.lower()
    return tweet

def tokenize(text):
    token = word_tokenize(text)
    return token

stop_word = stopwords.words('indonesian')
stop_word.extend({'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo',
                  'kalo', 'amp', 'biar', 'bikin', 'bilang',
                  'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                  'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                  'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'dm',
                  'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                  '&amp', 'yah', 'hallo', 'halo', 'hello', 'bgt', 'td',
                  'no', 'yaa', 'ae', 'kali', 'segera', 'rd', 'kak', 'gmn', 'min'})
list_stopwords = set(stop_word)
def remove_stopword(text):
    return [kata for kata in text
            if kata not in list_stopwords]

def stemming(text):
    stem_factory = StemmerFactory().create_stemmer()
    return [stem_factory.stem(text) for text in text]

def return_setence(text):
    return " ".join([teks for teks in text])

