import unidecode
import emoji
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

original_stopwords = stopwords.words('english')
additional_stopwords = ['none']
original_stopwords.extend(additional_stopwords)
stopwords = set(original_stopwords)

def clean_text(text):
    """
    This function takes in a text and performs a series of text cleaning,
    and outputs the cleaned text.
    """
    if type(text) == np.float:
        return ""
    temp = text.lower() # to lower case
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp) # remove @s
    temp = re.sub("#[A-Za-z0-9_]+","", temp) # remove hashtags
    temp = re.sub(r'http\S+', '', temp) # remove links
    temp = re.sub(r"www.\S+", "", temp) # remove links
    temp = re.sub(r'\n|[^a-zA-Z]', ' ', temp) # remove punctuation
    temp = temp.replace("\n", " ").split()
    temp = [w for w in temp if not w in stopwords] # remove stopwords
    temp = [w for w in temp if not w.isdigit()] # remove numbers
    temp = [unidecode.unidecode(w) for w in temp] # turn non-enlish letters to english letters
    temp = " ".join(word for word in temp)
    return temp

def lemmatize_words(text):
    """
    This function takes in a text and lemmatize the words using WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def construct_features(df):
    """
    This function takes in a dataframe and cleans up the texts, and constructs a series
    of text features.
    """
    #Perform label extraction on buckets
    ps = PorterStemmer()
    df["text_cleaned"] = [clean_text(t) for t in df["text"]]
    df['lemmatized_text'] = df['text'].apply(lambda text: lemmatize_words(text))    # lemmatization
    df['tokenized_text'] = df['lemmatized_text'].apply(word_tokenize)   # tokenization
    df['stemmed_text'] = df['text_cleaned'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()])) # stemming
    return df
    

