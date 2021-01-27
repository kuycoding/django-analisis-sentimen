import pandas as pd
import re
import numpy as np
import joblib
import pickle
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# konver label ke polaritas
def convert(polarity):
    if polarity == "positif":
        return 1
    elif polarity == "netral":
        return 0
    else:
        return -1

def convert1(polarity):
    if polarity == [1]:
        return 1
    elif polarity == [0]:
        return 0
    else:
        return -1

def convertToPolarity(polarity):
    if polarity == [1]:
        return 'positif'
    elif polarity == [0]:
        return 'netral'
    else:
        return 'negatif'

def convert2(polarity):
    if polarity == 1:
        return 'positif'
    elif polarity == 0:
        return 'netral'
    else:
        return 'negatif'