#!/usr/bin/env python
# coding: utf-8

# # Read CSV file using pandas

# In[4]:


import pandas as pd
pd.set_option('display.max_colwidth',100)

# Reading CSV File
data = pd.read_csv (r'Reviews.csv')

#Selected colums
data = data[['reviews.rating' , 'reviews.text' , 'reviews.title']]
data.head()


# # NLP Implementation

# # Remove punctuation from the text

# In[5]:


import string 
string.punctuation


# In[6]:


#Defining function
def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


# In[7]:


data['reviews.text_clean'] = data['reviews.text'].apply(lambda x: remove_punctuation(x))
data.head()


# In[8]:


data.to_csv('Reviewsclean.csv')


# # Tokenization

# In[9]:


import re

def tokenize(txt):
    tokens = re.split('\W+' , txt)
    return tokens

data['reviews.text_clean_tokenized'] = data['reviews.text_clean'].apply(lambda x: tokenize(x.lower()))
data.head()


# In[10]:


data.to_csv('Reviewstokenized.csv')


# # Removing Stopwords

# In[11]:


import nltk
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]


# In[12]:


#Defining function
def remove_stopwords(txt_tokenized):
    txt_clean  = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

data['reviews.text_no_sw'] = data['reviews.text_clean_tokenized'].apply(lambda x: remove_stopwords(x))
data.head()


# # Clear Data

# In[13]:


def clean_text(text):
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text


# In[14]:


data['reviews.text_no_sw'] = data['reviews.text'].apply(lambda x: clean_text(x.lower()))
data.head()


# In[15]:


data.to_csv('ReviewsNostopwords.csv')


# # NLTK implementation

# In[16]:


import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
dir(ps)


# In[17]:


def stemming(tokenized_text): 
    text = [ps.stem(word) for word in tokenized_text]
    return text
data['reviews.text_stemmed'] = data['reviews.text_no_sw'].apply(lambda x: stemming(x))
data.head()


# # Wordnet

# In[18]:


import nltk
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
dir(wn)


# # lemmatization

# In[21]:


#defining function
def lemmatization(token_txt):
    text = [wn.lemmatize(word) for word in token_txt]
    return text


# In[22]:


data['reviews.text_lemmatized'] = data['reviews.text_no_sw'].apply(lambda x: lemmatization(x))
data.head()


# In[23]:


data.to_csv('Reviewsstemming.csv')


# # Feature Engineering

# In[24]:


import string 
def punctuation_count(txt):
    count = sum([1 for c in txt if c in string.punctuation])
    return 100*count/len(txt)

data['punctuation_%'] = data['reviews.text'].apply(lambda x: punctuation_count(x))
data.head()


# # Sentimental Analysis

# In[25]:


# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
data["sentiments"] = data["reviews.text"].apply(lambda x: sid.polarity_scores(x))
data = pd.concat([data.drop(['sentiments'], axis=1), data['sentiments'].apply(pd.Series)], axis=1)


# In[26]:


# add number of characters column
data["R_chars"] = data["reviews.text"].apply(lambda x: len(x))

# add number of words column
data["R_words"] = data["reviews.text"].apply(lambda x: len(x.split(" ")))


# In[27]:


# highest positive sentiment reviews (with more than 5 words)
data[data["R_words"] >= 5].sort_values("pos", ascending = False)[["reviews.text", "pos"]].head(10)


# In[28]:


# lowest negative sentiment reviews (with more than 5 words)
data[data["R_words"] >= 5].sort_values("neg", ascending = False)[["reviews.text", "neg"]].head(10)


# In[29]:


data['sentiment'] = data['reviews.text'].apply(lambda x: sid.polarity_scores(x))
def convert(x):
    if x < 0:
        return "negative"
    elif x > .2:
        return "positive"
    else:
        return "neutral"
data['result'] = data['sentiment'].apply(lambda x:convert(x['compound']))


# In[30]:


data.to_csv('Senitmentalanalysis.csv')


# In[31]:


data.info()


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data.sentiment.value_counts()


# # Sentimental Analysis of ratings

# In[33]:


sentiment_count=data.groupby('sentiment').count()
plt.bar(sentiment_count.index.values, sentiment_count['reviews.text'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# In[ ]:


data.columns


# #Density plot of positive and negative reviews

# In[81]:


import seaborn as sns

for x in [1, 5]:
    subset = data[data['sentiment'] == x]
    
    # Draw the density plot
    if x == 5:
        label = "positive reviews"
    else:
        label = "Negative reviews"
    sns.distplot(subset['compound'], hist = False, label = label)


# # TF-IDF Vector 

# In[36]:


#Method1

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv = CountVectorizer()

cv1 = CountVectorizer(analyzer=clean_text)

X = cv1.fit_transform(data['reviews.text'])
print(X.shape)
#print(tfidf_vect.vocabulary_.keys())


# In[37]:


print(cv1.get_feature_names())


# In[38]:


data_sample = data[0:10]
tfidf2 = TfidfVectorizer(analyzer=clean_text)
X = tfidf2.fit_transform(data_sample['reviews.text'])
print(X.shape)


# In[39]:


#dataframe for tf-idf
df = pd.DataFrame(X.toarray(), columns=tfidf2.get_feature_names())
df.head(10)


# In[40]:


data.to_csv('Reviewsnlptrain.csv')


# In[41]:


data.isnull().sum()


# # Wordtovector

# In[42]:


# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["reviews.text_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = data["reviews.text_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
data = pd.concat([data, doc2vec_df], axis=1)


# In[43]:


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(data["reviews.text_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index =data.index
data = pd.concat([data, tfidf_df], axis=1)


# In[44]:


data.head()


# In[45]:


# show sentiment distribution
data["reviews.text"].value_counts(normalize = True)


# In[46]:


# show sentiment distribution
#data["sentiment"].value_counts(normalize = True)


# # word cloud of hotel reviews(clean data)

# In[47]:


# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 30, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 10)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
from tqdm import tqdm
show_wordcloud(data["reviews.text"])


# # naivy bayes model used for check accuracy

# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['reviews.text'])


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['reviews.rating'], test_size=0.3, random_state=1)


# In[58]:


from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[60]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['reviews.text'])


# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['reviews.rating'], test_size=0.3, random_state=123)


# In[62]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# # classification model for train and test data

# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
X_train, X_test, y_train, y_test = train_test_split(data[['reviews.text', 'punctuation_%']], data['result'], test_size=0.2)


# In[77]:


tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['reviews.text'])

tfidf_train = tfidf_vect_fit.transform(X_train['reviews.text'])
tfidf_test = tfidf_vect_fit.transform(X_test['reviews.text'])

X_train_vect = pd.concat([X_train[['punctuation_%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['punctuation_%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)

X_train_vect.head()


# In[78]:


from sklearn.feature_extraction.text import CountVectorizer
Reviewdata = CountVectorizer(analyzer=clean_text).fit_transform(data['reviews.text'])


# In[79]:


#Split the data into 80% training (X_train & y_train) and 20% testing (X_test & y_test) data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Reviewdata, data['reviews.rating'], test_size = 0.20, random_state = 0)


# In[80]:


Reviewdata.shape


# In[67]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[68]:


#Print the predictions
print(classifier.predict(X_train))

#Print the actual values
print(y_train.values)


# In[69]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))


# In[70]:


#Print the predictions
print('Predicted value: ',classifier.predict(X_test))

#Print Actual Label
print('Actual value: ',y_test.values)


# In[71]:


#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))


# # Naive bayes model for reviews testing

# In[72]:


from sklearn.naive_bayes import MultinomialNB
reviews_recommended = MultinomialNB().fit(tfidf_result, data['result'])


# In[73]:


print('expected:', reviews_recommended.predict(tfidf_result)[4])
print('recommended:', data.result[3])


# In[74]:


all_predictions = reviews_recommended.predict(tfidf_result)
print(all_predictions)


# In[75]:


from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(data['result'], all_predictions))
print('\n')
m_confusion_test = confusion_matrix(data['result'], all_predictions)
pd.DataFrame(data = m_confusion_test, columns = ['Predicted positive', 'Predicted negative' , 'Predicted neutral'],
            index = ['Actual positive', 'Actual Negavtive' , 'Actual neutral'])

