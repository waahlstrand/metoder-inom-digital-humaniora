import requests
import bs4
import urllib.request, urllib.error, urllib.parse
import re

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer

from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis.sklearn


def extract_text(html):
    
    html = bs4.BeautifulSoup(html, features="lxml")
    
    text = ''
    
    # Extract all environments with actual content
    for span in html.find_all('span'):
        
        text = text + span.string if span.string else text # Extract segments
        text = text + ' '                                  # Add a space between new segments
        
    return text

def query_riksdagen(topic='skatt', doc_type='mot', from_date='2010-01-01', tom_date='2020-01-01', limit=5, **kwargs):
    
    params = {'sok': topic,
            'doktyp': doc_type,
            'from': from_date,
            'tom': tom_date,
            'avd': 'dokument',
            'sort': 'rel',
            'sortorder': 'desc',
            'utformat': 'json',
            'a': 's'}
        
    next_page = '@nasta_sida'
    document_url = 'dokument_url_html'
    content = 'dokumentlista'
    documents = 'dokument'

    data = []
    
    # Using the web API for the Swedish parliaments
    api = 'http://data.riksdagen.se/dokumentlista/'
 
    has_next = True
    i = 0

    while has_next and (i < limit):

        # Get response from API
        response = (requests.get(api, params=params)
                            .json())

        has_next = next_page in response.get(content).keys()

        # Extract HTML from response
        # Loop all documents in list
        for document in response.get(content).get(documents):
            try:

                # Get the html page with the actual content
                html_url = 'http:'+document.get(document_url)
                html_response = urllib.request.urlopen(html_url).read()

                # Add text in document to data
                data.append(extract_text(html_response))

            except KeyError as ke:

                print(ke)
                
        api = response.get(content).get(next_page) if has_next else None
        i   = i+1
        
    return data


def TopicModel(pipeline, data, n_topics=5, **kwargs):

    processed = pipeline.fit_transform(data)

    lda = LDA(n_components=n_topics).fit(processed)
    return pyLDAvis.sklearn.prepare(lda, processed, pipeline[-1], **kwargs)


def remove_punctuation(x):

    # Remove regexed punctuation
    regex = '[:,\.!?+\']' 
    x = re.sub(regex, '', x)

    # Replace slash with simple space
    x = re.sub('[/\]\[\)\(]', ' ', x)

    # Replace dash with underscode
    x = re.sub('[-]', '_', x)

    return x


class StemTokenizeCount(CountVectorizer):

    def __init__(self, **kwargs):
        super().__init__(lowercase=False, analyzer="word", token_pattern="(?u)\w+(?:'\w+)?|[^\w\s]", **kwargs)


    def build_analyzer(self):
        stemmer = SnowballStemmer('swedish')
        analyzer = super(StemTokenizeCount, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

class TokenizeCount(CountVectorizer):

    def __init__(self, **kwargs):
        super().__init__(lowercase=False, analyzer="word", token_pattern="(?u)\w+(?:'\w+)?|[^\w\s]", **kwargs)

PunctuationRemover  = FunctionTransformer(lambda data: [remove_punctuation(d) for d in data])
LowerCaser          = FunctionTransformer(lambda data: [d.lower() for d in data])

