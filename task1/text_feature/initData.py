# -*- coding: utf-8 -*-
"""
__title__=""
__author__="ngc7293"
__mtime__="2020/6/3"
"""
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def process_sentence(sentence):
    sentence = BeautifulSoup(sentence,"lxml").get_text()
    sentence = re.sub(r"[^a-zA-Z]"," ",sentence)
    sentence = sentence.lower().split()
    # stop_words = set(stopwords.words("english"))
    stop_words = []
    res = [word for word in sentence if word not in stop_words]
    return " ".join(res)

