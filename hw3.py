import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

########## part 1: topic modelling ##########

df = pd.read_csv('ass3.csv', sep = ',')
# isolate reviews and remove null values in order to allow for tokenization
reviews = list(df['review'].dropna())

# tokenize reviews
tokens = [nltk.word_tokenize(i) for i in reviews]

# set lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# initialize empty list to store final tokens
finals = []
for x in tokens:
    # lemmatize words using 'verb' pos tag (determines the lemmas)
    new = [lemmatizer.lemmatize(token, pos = "v") for token in x if token.isalpha()]
    # remove stop words
    new = [token for token in new if not token in stopwords.words('english')]
    # append each review to finals list
    finals.append(new)

# merge each element in each sublist into one sentence
end = []
for x in finals:
    y = [' '.join(x)]
    end.append(y)

import itertools
# consolidate all reviews together into one list with 1000 elements (one per review)
final = list(itertools.chain.from_iterable(end))

# set vectorizer with min doc freq = 5 and 2-gram
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,2))
vectorizer.fit(final)
print(vectorizer.vocabulary_)
v = vectorizer.transform(final)
print(v.toarray())

# set all elements to lowercase
revs = []
for x in end:
    for y in x:
        y = y.lower()
        revs.append(y)

print("pre-processing complete!")

# LDA using sklearn
vectorizer2 = CountVectorizer(stop_words = 'english')
v2 = vectorizer2.fit_transform(revs)
terms = vectorizer2.get_feature_names()
from sklearn.decomposition import LatentDirichletAllocation
# use lda model to extract six topics
lda = LatentDirichletAllocation(n_components = 6, learning_method='online',
                                learning_decay=0.4, random_state=0).fit(v2)
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d: " % (topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[:-4-1:-1]]))
    print()

# Topic 0: movie about war and death
# war human kill film
# Topic 1: delicious restaurant
# eat good delicious taste
# Topic 2: probably movie with these characters/actors
# xixia xiaopeng barbara lejia
# Topic 3: probably movie about love
# quot people love like
# Topic 4: definitely movie/film
# film movie audience character
# Topic 5: drama film
# quot film drama character

output = lda.fit_transform(v2)
topicnames = ["Topic" + str(x) for x in range(lda.n_components)] # set names for topics
docnames = ["Document" + str(x) for x in range(len(df)-1)] # set doc names
# build dataframe of topicnames and docnames
doc_topic = pd.DataFrame(np.round(output,2), columns=topicnames, index=docnames)
dominant_topic = np.argmax(doc_topic.values, axis=1)
doc_topic['dominant_topic'] = dominant_topic
# find dominant topics for first 10 restaurant reviews
doc_topic.iloc[0:10]
# find dominant topics for first 10 movie reviews
doc_topic.iloc[500:510]

# build dataframe with topic keywords for each review
keywords = pd.DataFrame(lda.components_)
keywords.columns = terms
keywords.index = topicnames
keywords.head(10)

# show top 5 keywords per topic
key = np.array(terms)
topic_keywords = []

# iterate through topic weights and append them to topic_keywords list
for weights in lda.components_:
    top_key_locs = (-weights).argsort()[:5]
    topic_keywords.append(key.take(top_key_locs))

top_keys = pd.DataFrame(topic_keywords)
top_keys.index = ['Topic '+str(x) for x in range(top_keys.shape[0])]
top_keys.columns = ['Word '+str(x) for x in range(top_keys.shape[1])]
top_keys

#            Word 0    Word 1   Word 2       Word 3     Word 4
# Topic 0        li   dynasty    xixia      barbara       song # perhaps a chinese movie
# Topic 1      loki  raytheon   estate          sun     wukong # another movie (thor/avengers)
# Topic 2      kung      bank  martial         arts      train # martial arts/action movie
# Topic 3  xiaopeng     plato    japan  philosopher  diplomacy # philosophical movie
# Topic 4       eat      good     like    delicious      taste # restaurant
# Topic 5      quot      film   people         love       time # film about life and love

########## part 2: classification ##########

# split into test and training set
train_reviews = pd.concat([df.loc[:399,:], df.loc[500:899,:]]).dropna()
test_reviews = pd.concat([df.loc[400:499,], df.loc[900:,:]]).dropna()

x_train = train_reviews['review'].tolist()
x_test = test_reviews['review'].tolist()

y_train = train_reviews['label'].tolist()
y_test = test_reviews['label'].tolist()

# tokenize reviews
train_tokens = [nltk.word_tokenize(i) for i in x_train]
test_tokens = [nltk.word_tokenize(i) for i in x_test]

# use lemmatizer from q1
# initialize empty lists to store final tokens
final_train, final_test = [], []
for x in train_tokens:
    # lemmatize words using 'verb' pos tag (determines the lemmas)
    new = [lemmatizer.lemmatize(token, pos="v") for token in x if token.isalpha()]
    # remove stop words
    new = [token for token in new if not token in stopwords.words('english')]
    # append each review to finals list
    final_train.append(new)

for x in test_tokens:
    # lemmatize words using 'verb' pos tag (determines the lemmas)
    new = [lemmatizer.lemmatize(token, pos="v") for token in x if token.isalpha()]
    # remove stop words
    new = [token for token in new if not token in stopwords.words('english')]
    # append each review to finals list
    final_test.append(new)

# initialize empty lists to store final sentences
train_end, test_end = [], []
# merge each element in each sublist into one sentence
for x in final_train:
    y = [' '.join(x)]
    train_end.append(y)

for x in final_test:
    y = [' '.join(x)]
    test_end.append(y)

train = list(itertools.chain.from_iterable(train_end))
test = list(itertools.chain.from_iterable(test_end))

# set vectorizer with min doc freq = 5 and 2-gram
vectorizer2 = TfidfVectorizer(min_df=5, ngram_range=(1,2))
vectorizer2.fit(train)
print(vectorizer2.vocabulary_)

v1 = vectorizer2.transform(train)
print(v1.toarray())

v2 = vectorizer2.transform(test)
print(v2.toarray())

# set all elements to lowercase
revs_train, revs_test = [], []
for x in train_end:
    for y in x:
        y = y.lower()
        revs_train.append(y)

for x in test_end:
    for y in x:
        y = y.lower()
        revs_test.append(y)

print('pre-processing complete!')

# logit model
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()

# train
logit.fit(v1, y_train)
y_pred_logit = logit.predict(v2)

# evaluate
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(y_test, y_pred_logit)
print("Logit Model Accuracy: {:.2f}%".format(acc_logit*100)) # 99.00%

# naive bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# train
nb.fit(v1, y_train)
y_pred_nb = nb.predict(v2)

# evaluate
acc_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Model Accuracy:: {:.2f}%".format(acc_nb*100)) # 98.50%

# random model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, bootstrap=True, random_state=0)

# train
rf.fit(v1, y_train)
y_pred_rf = rf.predict(v2)

# evaluate
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_rf*100)) # 94.50%

# SVM
from sklearn.svm import LinearSVC
svm = LinearSVC()

# train
svm.fit(v1, y_train)
y_pred_svm = svm.predict(v2)

# evaluate
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Model Accuracy: {:.2f}%".format(acc_svm*100)) # 99.00%

# neural network and deep learning
from sklearn.neural_network import MLPClassifier
# set single layer with 4 nodes
nn = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,1))
# set 2 nodes and a depth of 2 layers
dl = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(2,2))
# set 4 nodes and a depth of 3 layers
dl2 = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(4,3))
# set 2 nodes and a depth of 3 layers
dl3 = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(2,3))
# set 4 nodes and a depth of 2 layers
dl4 = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes=(4,2))

# train
nn.fit(v1, y_train)
y_pred_nn = nn.predict(v2)
dl.fit(v1, y_train)
y_pred_dl = dl.predict(v2)
dl2.fit(v1, y_train)
y_pred_dl2 = dl2.predict(v2)
dl3.fit(v1, y_train)
y_pred_dl3 = dl3.predict(v2)
dl4.fit(v1, y_train)
y_pred_dl4 = dl4.predict(v2)

# evaluate
acc_nn = accuracy_score(y_test, y_pred_nn)
acc_dl = accuracy_score(y_test, y_pred_dl)
acc_dl2 = accuracy_score(y_test, y_pred_dl2)
acc_dl3 = accuracy_score(y_test, y_pred_dl3)
acc_dl4 = accuracy_score(y_test, y_pred_dl4)
print("NN Model Accuracy: {:.2f}%".format(acc_nn*100)) # 99.00%
print("DL Model Accuracy: {:.2f}%".format(acc_dl*100)) # 99.00%
print("DL Model #2 Accuracy: {:.2f}%".format(acc_dl2*100)) # 50.00%
print("DL Model #3 Accuracy: {:.2f}%".format(acc_dl3*100)) # 50.00%
print("DL Model #4 Accuracy: {:.2f}%".format(acc_dl4*100)) # 99.00%
