import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import re

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer


from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import gensim
import gensim.downloader

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Data:

        def __init__(self, text) -> None:
            
            self.text = text

        def remove_punctuation(self):

                # Remove regexed punctuation
                regex = '[:,\.!?+\']' 
                self.text = re.sub(regex, '', self.text)

                # Replace slash with simple space
                self.text = re.sub('[/\]\[\)\(]', ' ', self.text)

                # Replace dash with underscode
                self.text = re.sub('[-]', '_', self.text)

                return self 

        def tokenize(self):

                self.text = word_tokenize(self.text)

                return self

        def to_lower_case(self):

                self.text = [token.lower() for token in self.text]

                return self

        def lemmatize(self):

                lemmatizer = WordNetLemmatizer()
                self.text = [lemmatizer.lemmatize(token) for token in self.text]

                return self

        def remove_stopwords(self, custom=[]):

                stopwords = nltk.corpus.stopwords.words('english')
                stopwords.extend(custom)

                self.text = [token for token in self.text if token not in stopwords]

                return self

        def add_compounds(self, compounds):

                compounder = MWETokenizer(compounds)
                self.text = compounder.tokenize(self.text)

                return self

        def remove_low_frequency_tokens(self, frequency=0.01):

                counts = Counter(self.text)
                total  = sum(counts.values())

                counts = {k: v/total for k,v in counts.items()}

                self.text = [token for token in self.text if counts[token] > frequency]

                return self

        def ordinal_encoding(self):
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(self.text)

                return lambda key: encoder.transform(key)

        def count_encoding(self):
                # vectorizer = CountVectorizer(vocabulary=set(self.text))
                # encoded = vectorizer.fit_transform(self.text)

                encoded = Counter(self.text)
        
                return lambda key: encoded[key]

        def to_ngrams(self, n):

                return nltk.ngrams(self.text, n)

        def __repr__(self):

                return repr(self.text)
                

def example():

        text = """The definition of the digital humanities is being continually formulated by scholars/practitioners. Since the field is constantly growing and changing, specific definitions can 
        quickly become outdated or unnecessarily limit future potential.[4] The second volume of Debates 
        in the Digital Humanities (2016) acknowledges the difficulty in defining the field: 'Along with the 
        digital archives, quantitative analyses, and tool-building projects that once characterized the 
        field, DH now encompasses a wide range of methods and practices: visualizations of large image 
        sets, 3D modeling of historical artifacts, 'born digital' dissertations, hashtag activism and the 
        analysis thereof, alternate reality games, mobile makerspaces, and more. In what has been called 
        'big tent' DH, it can at times be difficult to determine with any specificity what, precisely, digital 
        humanities work entails.'[5] Historically, the digital humanities developed out of humanities computing 
        and has become associated with other fields, such as humanistic computing, social computing, and media studies. 
        In concrete terms, the digital humanities embraces a variety of topics, from curating online collections 
        of primary sources (primarily textual) to the data mining of large cultural data sets to topic modeling. 
        Digital humanities incorporates both digitized (remediated) and born-digital materials and combines the 
        methodologies from traditional humanities disciplines (such as rhetoric, history, philosophy, linguistics, 
        literature, art, archaeology, music, and cultural studies) and social sciences,[6] with tools provided by 
        computing (such as hypertext, hypermedia, data visualisation, information retrieval, data mining, 
        statistics, text mining, digital mapping), and digital publishing. Related subfields of digital 
        humanities have emerged like software studies, platform studies, and critical code studies. Fields 
        that parallel the digital humanities include new media studies and information science as well as 
        media theory of composition, game studies, particularly in areas related to digital humanities project 
        design and production, and cultural analytics."""

        return Data(text)

def embeddings_example(seed=123):

        print("Downloading, this will take some time...")
        embedding = gensim.downloader.load('glove-twitter-25')
        print("- Done! Initializing...")

        data = (example()
              .remove_punctuation()
              .tokenize()
              .to_lower_case()
              .lemmatize()
              .remove_stopwords(['ha','1', '2', '3', '4', '5', '6', '7', '8' , '2016', 
                                 'tool_building', '3d', 'makerspaces', 'remediated', 'born_digital', 'subfields'])
       )

        pca = PCA(n_components=2)
        X = ['data', 'digital', 'humanities', 'classics']
        Z = pca.fit_transform(embedding[data.text])

        np.random.seed(seed)
        fig, ax = plt.subplots(figsize=(15,15))
        ax.scatter(Z[:,0], Z[:,1], alpha=0)

        idxs = np.random.choice(range(len(data.text)), 30)

        for i in idxs:
                ax.annotate(data.text[i], (Z[i,0], Z[i,1]))

        ax.set_xlabel('first dimension')
        ax.set_ylabel('second dimension')
        return fig, ax