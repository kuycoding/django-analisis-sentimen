# -*- encoding: utf-8 -*-
"""
MIT License
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views
from django.conf.urls import handler404, handler500

urlpatterns = [
    # Matches any html file - to be used for gentella
    # Avoid using your .html in your resources.
    # Or create a separate django app.
    re_path(r'^.*\.html', views.pages, name='pages'),

    # The home page
    path('', views.index, name='home'),
    path('scraping', views.scraping, name='scraping'),
    path('scrape', views.scrape, name='scrape'),
    path('scrape/download', views.download_scrape, name="download"),
    path('preprocessing', views.preprocessing, name="preprocessing"),
    path('preprocessing/download', views.download_preprocessing),
    path('klasifikasi', views.klasifikasi, name="klasifikasi"),
    path('visualisasi', views.visualisasi, name="visualisasi"),
    path('model_predic', views.model_predic, name="model predic"),
    path('model/download', views.download_model),
    path('visualize', views.visualize, name="visualize")

]

handler404 = 'app.views.error_404'
handler500 = 'app.views.error_500'
