# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect, render_to_response
from django.template import loader, RequestContext
from django.http import HttpResponse
from django import template
from django.core.exceptions import ObjectDoesNotExist
from . import Scraping_Twitter
from . import Preprocessing_Twitter
from . import Klasifikasi_NB
import base64
import pandas as pd
import tweepy
import csv
import json
import numpy as np
import joblib
import pickle
import time
import io
import os
import re
import matplotlib
matplotlib.use('Agg')
import multiprocessing
multiprocessing.set_start_method('spawn')
import plotly.express as px
import plotly.offline as opy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datetime import datetime

consumer_key = "x9zuaTU1HbDErb9amFHGyY2kY"
consumer_secret = "6tZlXskc8QD9f8j9FthqZzLFHN6E2aH1nINfMlqJu2pWg3MsEc"
access_token = "1278541122512408576-ClQSTF1hM173Maz8Sy6n109zJxiTS4"
access_token_secret = "JqDkBY5iamljZPcrs0ayv0I26UoKa3TEnxZMGZT7EhXoz"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def error_404(request, exception=None):
    return render(request,'core/templates/page-404.html')

def error_500(request):
    return render(request,'core/templates/page-500.html')

@login_required(login_url="/login/")
def index(request):
    
    context = {}
    context['segment'] = 'index'
    keywords = "indihome"
    filterKey = " -filter:retweets"
    counts = "10"
    tweet = context['tweet'] = tweepy.Cursor(api.search, q=keywords + filterKey, lang="in", tweet_mode='extended').items(int(counts))
    json_data = [r._json for r in tweet]
    df = pd.json_normalize(json_data)
    df['keyword'] = keywords
    df['created_at'] = pd.to_datetime(df['created_at'], format='%Y%m%d', errors='ignore')
    df.rename(columns={'created_at':'date','user.profile_image_url':'photo','user.screen_name':'username','full_text':'tweet'}, inplace=True)
    print(df)

    # tabel 
    json_records = df.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['tweet'] = data

    # sentimen analisis
    with io.open('app/model_analyze1.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with io.open('app/count_vectorize1.pkl', 'rb') as h:
        cvec = pickle.load(h)
   
    with io.open('app/tfidf_tranform1.pkl', 'rb') as t:
        tfidf = pickle.load(t)

    tweet = []
    clean_text = []
    clas = []
    for i, line in df.iterrows():
        isi = line[3]
        print(isi)

        #  preprocessing
        clean_text_isi = Preprocessing_Twitter.clean(isi)
        print(clean_text_isi)
        remove_number_isi = Preprocessing_Twitter.remove_number(clean_text_isi)
        remove_punc_isi = Preprocessing_Twitter.remove_punctuation(remove_number_isi)
        remove_single_char_isi = Preprocessing_Twitter.remove_singl_char(remove_punc_isi)
        remove_whitespace_LT_isi = Preprocessing_Twitter.remove_whitespace_LT(remove_single_char_isi)
        remove_whitespace_multiple_isi = Preprocessing_Twitter.remove_whitespace_multiple(remove_whitespace_LT_isi)
        casefolding_isi = Preprocessing_Twitter.casefolding(remove_whitespace_multiple_isi)
        tokenize_isi = Preprocessing_Twitter.tokenize(casefolding_isi)
        stopword_isi = Preprocessing_Twitter.remove_stopword(tokenize_isi)
        stemming_isi = Preprocessing_Twitter.stemming(stopword_isi)
        return_setence_isi = Preprocessing_Twitter.return_setence(stemming_isi)
        print(return_setence_isi)

        # # transform cvector & tfidf
        transform_cvec = cvec.transform([return_setence_isi])
        transform_tfidf = tfidf.transform(transform_cvec)
        print(transform_tfidf)
        # predict start
        predic_result = model.predict(transform_tfidf)
        print(predic_result)

        tweet.append(isi)
        clean_text.append(return_setence_isi)
        clas.append(predic_result)

    df['tweet'] = tweet
    df['clean_text'] = clean_text
    df['class'] = clas
    df['class'] = df['class'].apply(Klasifikasi_NB.convert1)
    df['polaritas'] = df['class'].apply(Klasifikasi_NB.convert2)

    sentiment_count = df['class'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

    # random sentiment tweet
    random_pos = df.loc[df['polaritas'] == 'positif'].sample(n=2, replace = True)
    context['pos_uname'] = random_pos['username'].to_string(index=False)
    context['pos_date'] = random_pos['date'].to_string(index=False)
    context['pos_tweet'] = random_pos['tweet'].to_string(index=False)
    json_records = random_pos.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['p'] = data 

    random_net = df.loc[df['polaritas'] == 'netral'].sample(n=2, replace = True)
    context['net_uname'] = random_net['username'].to_string(index=False)
    context['net_date'] = random_net['date'].to_string(index=False)
    context['net_tweet'] = random_net['tweet'].to_string(index=False)
    json_records = random_net.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['net'] = data 

    random_neg = df.loc[df['polaritas'] == 'negatif'].sample(n=2, replace = True)
    context['neg_uname'] = random_neg['username'].to_string(index=False)
    context['neg_date'] = random_neg['date'].to_string(index=False)
    context['neg_tweet'] = random_neg['tweet'].to_string(index=False)
    json_records = random_neg.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['neg'] = data 

    # bar
    graph = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
    graph.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_bar = opy.plot(graph, output_type='div')
    context['bar'] = div_bar

    # pie
    fig_2 = px.pie(sentiment_count, values='Tweets', names='Sentiment')
    fig_2.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(font=dict(family='sans-serif')))
    div_pie = opy.plot(fig_2, output_type='div')
    context['pie'] = div_pie

    # bar-bar
    choice_data = df
    print(choice_data)
    fig_0 = px.histogram(choice_data, x='keyword', y='polaritas',
        histfunc='count', color='polaritas',
        facet_col='polaritas', labels={'polaritas': 'tweets'},
        height=500)
    fig_0.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_fig0 = opy.plot(fig_0, output_type='div')
    context['fig0'] = div_fig0
    
    # # wordcloud
    # w = ' '.join([text for text in df['clean_text'][df['polaritas'] == -1]])
    # wordcloud = WordCloud(stopwords=STOPWORDS, width=800, height=500, background_color="black", random_state=21, max_font_size=110).generate(w)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # buffer = io.BytesIO()
    # plt.savefig(buffer, format='png', facecolor='k', bbox_inches='tight')
    # buffer.seek(0)
    # strings = base64.b64encode(buffer.getvalue())
    # buffer.close()
    # image_64 = strings.decode('utf-8')
    # context['wc_1'] = image_64

    html_template = loader.get_template( 'index.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))
        
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def scraping(request):
    context = {}
    context['segment'] = 'scraping'

    html_template = loader.get_template( 'scraping.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def scrape(request):
    context = {}
    context['segment'] = 'scraping'
    
    if request.method == 'POST':
        print("Ambil data")
        global keywords, counts
        keywords = context['keyword'] = request.POST['keyword']
        counts = context['count'] = request.POST['count']
        if counts is None:
            counts == 1
    else:
        print("tidak apa-apa")

    filterKey = " -filter:retweets"
    Scraping_Twitter.authenticate(consumer_key, consumer_secret, access_token, access_token_secret)
    try:
        global key, df
        key = tweepy.Cursor(api.search, q=keywords + filterKey, lang="in", tweet_mode='extended').items(int(counts))
        json_data = [r._json for r in key]
        df = pd.json_normalize(json_data)
        df['keyword'] = keywords
        df['created_at'] = df['created_at'].astype('datetime64[ns]')
        df.rename(columns={'cre8ated_at':'date','user.profile_image_url':'photo','user.screen_name':'username','full_text':'tweet'}, inplace=True)
        print(df)

        global tw
        tw = context['tw'] = tweepy.Cursor(api.search, q=keywords + filterKey, lang="in", tweet_mode='extended').items(int(counts))
    
    except tweepy.TweepError as e:
        print("Tweepy error: {}".format(e), e.reason)
        context['tweepy_error'] = "Tweepy error: {}".format(e)
        time.sleep(3000)
    except StopIteration:
        pass
    

    html_template = loader.get_template( 'scraping.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def download_scrape(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dataScrape.csv"'
    
    header = ['keyword','date', 'photo', 'username', 'tweet']
    df[header].to_csv(path_or_buf=response,index=False)    
    return response

@login_required(login_url="/login/")
def preprocessing(request):
    context = {}
    context['segment'] = 'Preprocessing'

    if request.method == 'POST':
        testF = request.FILES.get(u'csvF')
        if testF:
            global df
            df = pd.read_csv(testF)
            context['df'] = df
            df['text'] = df['tweet'].astype(str)
            print(df)

            # clean = Preprocessing.clean(df['text'])
            df['cleans'] = df['text'].apply(lambda x: Preprocessing_Twitter.clean(x))
            df['cleans'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.remove_number(x))
            df['cleans'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.remove_punctuation(x))
            df['cleans'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.remove_singl_char(x))
            df['cleans'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.remove_whitespace_LT(x))
            df['cleans'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.remove_whitespace_multiple(x))
            df['caseFolding'] = df['cleans'].apply(lambda x: Preprocessing_Twitter.casefolding(x))
            df['tokenize'] = df['caseFolding'].apply(lambda x: Preprocessing_Twitter.tokenize(x))
            df['stopword'] = df['tokenize'].apply(lambda x: Preprocessing_Twitter.remove_stopword(x))
            df['stemming'] = df['stopword'].apply(lambda x: Preprocessing_Twitter.stemming(x))
            df['clean_text'] = df['stemming'].apply(lambda x: Preprocessing_Twitter.return_setence(x))

            json_records = df.reset_index().to_json(orient = 'records')
            data = []
            data = json.loads(json_records)
            context['d'] = data 
        else:
            print(request, f'No file to process! Please upload a file to process.')

    html_template = loader.get_template( 'preprocessing.html' )
    return HttpResponse(html_template.render(context, request))

def download_preprocessing(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="dataClean.csv"'

    header = ['keyword','date','photo','username','tweet','clean_text']
    df[header].to_csv(path_or_buf=response,index=False)
    
    return response

@login_required(login_url="/login/")
def klasifikasi(request):
    context = {}
    context['segment'] = 'Klasifikasi'

    if request.method == 'POST':
        testF = request.FILES.get(u'csvF')
        if testF:
            global df
            df = pd.read_csv(testF)
            context['df'] = df
            df['text'] = df['clean_text'].astype(str)
            
            # ubah ke polaritas
            df['polaritas'] = df['class'].apply(Klasifikasi_NB.convert)
            print("total polaritas ")
            pol = df['polaritas'].value_counts().to_frame()
            json_records = pol.reset_index().to_json(orient = 'records')
            jml = []
            jml = json.loads(json_records)
            context['jml'] = jml
            print(jml)
            x = df['text']
            y = df['polaritas']

            # vectorize
            bow_transform = CountVectorizer()
            print(df['text'].shape)
            context['vector'] = df['text'].shape

            x = bow_transform.fit_transform(df['text'])
            context['xtoarray'] = x.toarray()
            context['xshape'] = x.shape
            context['xnnz'] = x.nnz
            print(x.toarray())
            print("shape of sparse matrix ", x.shape)
            print("ammount of Non-Zero occurence ", x.nnz)

            # tfidf transform
            tf_transform = TfidfTransformer(use_idf=False).fit(x)
            x = tf_transform.transform(x)
            print(x.shape)
            context['xshape'] = x.shape

            # save TF-IDF
            # filename = 'tfidf_tranform1.pkl'
            # pickle.dump(tf_transform, open(filename, 'wb'))

            density = (100.0 * x.nnz / (x.shape[0] * x.shape[1]))
            print("density: {}".format((density)))
            context['density'] = "density: {}".format((density))

            # splite data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            # clasification data
            nb = MultinomialNB()
            nb.fit(x_train, y_train)
            predic = nb.predict(x_test)
            print(classification_report(y_test, predic))

            report = classification_report(y_test, predic, output_dict=True)
            dfreport = pd.DataFrame(report).transpose()
            dfreport.rename(columns={'f1-score':'f1_score'}, inplace=True)
            json_records = dfreport.reset_index().to_json(orient = 'records')
            data = []
            data = json.loads(json_records)
            context['klasifikasi'] = data
            # akurasi
            global accuracy
            accuracy = context['accuracy'] = accuracy_score(y_test, predic)*100

            json_records = df.reset_index().to_json(orient = 'records')
            data = []
            data = json.loads(json_records)
            context['k'] = data 

            # Confusion matrix
            unique_label = np.unique([y_test, predic])
            cm = confusion_matrix(y_test, predic,labels=unique_label)
            matrix = pd.DataFrame(cm, index=['{:}'.format(x) for x in unique_label],
                                columns=['{:}'.format(x) for x in unique_label])
            matrix.rename(
                columns={'1':'pos','0':'net','-1':'neg'},
                index={'1':'positif','0':'netral','-1':'negatif'}, inplace=True)
            
            print(matrix)
            json_records = matrix.reset_index().to_json(orient = 'records')
            data = []
            data = json.loads(json_records)
            context['matrix'] = data 
    else:
        print(request, f'No file to process! Please upload a file to process.')


    html_template = loader.get_template( 'klasifikasi.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def visualize(request):
    context = {}
    context['segment'] = 'Visualisasi'
    
    global sentiment_count
    sentiment_count = df['class'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})
    # random sentiment tweet
    
    random_pos = df.loc[df['class'] == 'positif'].sample(n=2)
    context['pos_uname'] = random_pos['username'].to_string(index=False)
    context['pos_date'] = random_pos['date'].to_string(index=False)
    context['pos_tweet'] = random_pos['tweet'].to_string(index=False)
    json_records = random_pos.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['p'] = data 

    random_net = df.loc[df['class'] == 'netral'].sample(n=2)
    context['net_uname'] = random_net['username'].to_string(index=False)
    context['net_date'] = random_net['date'].to_string(index=False)
    context['net_tweet'] = random_net['tweet'].to_string(index=False)
    json_records = random_net.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['net'] = data 

    random_neg = df.loc[df['class'] == 'negatif'].sample(n=2)
    context['neg_uname'] = random_neg['username'].to_string(index=False)
    context['neg_date'] = random_neg['date'].to_string(index=False)
    context['neg_tweet'] = random_neg['tweet'].to_string(index=False)
    json_records = random_neg.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['neg'] = data 

    # bar
    graph = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
    graph.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_bar = opy.plot(graph, output_type='div')
    context['bar'] = div_bar

    # pie
    fig_2 = px.pie(sentiment_count, values='Tweets', names='Sentiment')
    fig_2.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(font=dict(family='sans-serif')))
    div_pie = opy.plot(fig_2, output_type='div')
    context['pie'] = div_pie

    # bar-bar
    choice_data = df
    print(choice_data)
    fig_0 = px.histogram(choice_data, x='keyword', y='class',
        histfunc='count', color='class',
        facet_col='class', labels={'class': 'tweets'},
        height=500)
    fig_0.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_fig0 = opy.plot(fig_0, output_type='div')
    context['fig0'] = div_fig0
    
    # wordcloud
    w = ' '.join([text for text in df['clean_text'][df['polaritas'] == -1]])
    wordcloud = WordCloud(stopwords=STOPWORDS, width=800, height=500, background_color="black", random_state=21, max_font_size=110).generate(w)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor='k', bbox_inches='tight')
    buffer.seek(0)
    strings = base64.b64encode(buffer.getvalue())
    buffer.close()
    image_64 = strings.decode('utf-8')
    context['wc_1'] = image_64

    html_template = loader.get_template( 'visualize.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def model_predic(request):
    context = {}
    context['segment'] = 'Model Predic'

    if request.method == 'POST':
        testF = request.FILES.get(u'csvF')
        if testF:
            global df
            df = pd.read_csv(testF,encoding='utf-8', dtype=str)
            context['df'] = df
            

            with io.open('app/model_analyze1.pkl', 'rb') as handle:
                model = pickle.load(handle)
            
            with io.open('app/count_vectorize1.pkl', 'rb') as h:
                cvec = pickle.load(h)
               
            with io.open('app/tfidf_tranform1.pkl', 'rb') as t:
                tfidf = pickle.load(t)

            tweet = []
            clean_text = []
            clas = []
            for i, line in df.iterrows():
                isi = line[4]
                print(isi)
            
                #  preprocessing
                clean_text_isi = Preprocessing_Twitter.clean(isi)
                print(clean_text_isi)
                remove_number_isi = Preprocessing_Twitter.remove_number(clean_text_isi)
                remove_punc_isi = Preprocessing_Twitter.remove_punctuation(remove_number_isi)
                remove_single_char_isi = Preprocessing_Twitter.remove_singl_char(remove_punc_isi)
                remove_whitespace_LT_isi = Preprocessing_Twitter.remove_whitespace_LT(remove_single_char_isi)
                remove_whitespace_multiple_isi = Preprocessing_Twitter.remove_whitespace_multiple(remove_whitespace_LT_isi)
                casefolding_isi = Preprocessing_Twitter.casefolding(remove_whitespace_multiple_isi)
                tokenize_isi = Preprocessing_Twitter.tokenize(casefolding_isi)
                stopword_isi = Preprocessing_Twitter.remove_stopword(tokenize_isi)
                stemming_isi = Preprocessing_Twitter.stemming(stopword_isi)
                return_setence_isi = Preprocessing_Twitter.return_setence(stemming_isi)
                print(return_setence_isi)

                # # transform cvector & tfidf
                transform_cvec = cvec.transform([return_setence_isi])
                transform_tfidf = tfidf.transform(transform_cvec)
                print(transform_tfidf)
                # predict start
                predic_result = model.predict(transform_tfidf)
                print(predic_result)

                tweet.append(isi)
                clean_text.append(return_setence_isi)
                clas.append(predic_result)

            df['tweet'] = tweet
            df['clean_text'] = clean_text
            df['class'] = clas
            df['class'] = df['class'].apply(Klasifikasi_NB.convertToPolarity)
            result = pd.concat([df['keyword'],df['date'],df['photo'],df['username'],df['tweet'],df['clean_text'],df['class'] ], axis=1, join='inner')
            # to json
            json_records = result.reset_index().to_json(orient = 'records')
            data = []
            data = json.loads(json_records)
            print(df)
            context['predic'] = data 
            
    else:
        print(request, f'No file to process! Please upload a file to process.')

    html_template = loader.get_template( 'model_predic.html' )
    return HttpResponse(html_template.render(context, request))

def visualisasi(request):
    context = {}
    context['segment'] = 'Visualisasi'

    df['polaritas'] = df['class'].astype(str)
    df['class'] = df['class'].apply(Klasifikasi_NB.convert)
    global sentiment_count
    sentiment_count = df['polaritas'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})
    # random sentiment tweet
    
    random_pos = df.loc[df['polaritas'] == 'positif'].sample(n=2, replace = True)
    context['pos_uname'] = random_pos['username'].to_string(index=False)
    context['pos_date'] = random_pos['date'].to_string(index=False)
    context['pos_tweet'] = random_pos['tweet'].to_string(index=False)
    json_records = random_pos.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['p'] = data 

    random_net = df.loc[df['polaritas'] == 'netral'].sample(n=2, replace = True)
    context['net_uname'] = random_net['username'].to_string(index=False)
    context['net_date'] = random_net['date'].to_string(index=False)
    context['net_tweet'] = random_net['tweet'].to_string(index=False)
    json_records = random_net.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['net'] = data 

    random_neg = df.loc[df['polaritas'] == 'negatif'].sample(n=2, replace = True)
    context['neg_uname'] = random_neg['username'].to_string(index=False)
    context['neg_date'] = random_neg['date'].to_string(index=False)
    context['neg_tweet'] = random_neg['tweet'].to_string(index=False)
    json_records = random_neg.reset_index().to_json(orient = 'records')
    data = []
    data = json.loads(json_records)
    context['neg'] = data 

    # bar
    graph = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
    graph.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_bar = opy.plot(graph, output_type='div')
    context['bar'] = div_bar

    # pie
    fig_2 = px.pie(sentiment_count, values='Tweets', names='Sentiment')
    fig_2.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hoverlabel=dict(font=dict(family='sans-serif')))
    div_pie = opy.plot(fig_2, output_type='div')
    context['pie'] = div_pie
    
    # bar-bar
    choice_data = df
    print(choice_data)
    fig_0 = px.histogram(choice_data, x='keyword', y='polaritas',
        histfunc='count', color='polaritas',
        facet_col='polaritas', labels={'polaritas': 'tweets'},
        height=500)
    fig_0.update_layout(font_color="#e6e6e6", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    div_fig0 = opy.plot(fig_0, output_type='div')
    context['fig0'] = div_fig0
    
    # wordcloud
    w = ' '.join([text for text in df['clean_text'][df['polaritas'] == 'negatif']])
    wordcloud = WordCloud(stopwords=STOPWORDS, width=800, height=500, background_color="black", random_state=21, max_font_size=110).generate(w)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', facecolor='k', bbox_inches='tight')
    buffer.seek(0)
    strings = base64.b64encode(buffer.getvalue())
    buffer.close()
    image_64 = strings.decode('utf-8')
    context['wc_1'] = image_64

    html_template = loader.get_template( 'visualize.html' )
    return HttpResponse(html_template.render(context, request))

def download_model(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="modelPredic.csv"'

    header = ['keyword','date','photo','username','tweet','clean_text','class']
    df[header].to_csv(path_or_buf=response,index=False)
    
    return response
    