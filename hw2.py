import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('ass2.csv', header = None)
reviews = list(df[1])

########## step 1 ##########

# tokenize each review based to produce nested list of tokenized words per review
tokens = [nltk.word_tokenize(i) for i in reviews] # 100 lists (each represents 1 review)

########## step 2-3 ##########

# set lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# initialize empty list to store final tokens for conversion
finals = []
for x in tokens:
    # lemmatize words using 'verb' part-of-speech tag to find lemmas
    # isalpha() function used to remove punctuation and other symbols
    new_tokens = [lemmatizer.lemmatize(token, pos="v") for token in x if token.isalpha()]
    # replace variable new_tokens with stopwords removed
    new_tokens = [token for token in new_tokens if not token in stopwords.words('english') if token.isalpha()]
    # appened each review to finals list
    finals.append(new_tokens) # list of lists

# merge each element in each sublist to one sentence
# initialize empty list
final = []
for x in finals:
    y = [' '.join(x)]
    final.append(y)

import itertools
# convert into one sentence with 100 elements (each representing one review)
new = list(itertools.chain.from_iterable(final))

########## step 4 ##########

# set vectorizer to convert reviews to TDIDF with min doc freq per term = 3 and 2-gram
vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1,2))
vectorizer.fit(new)
print(vectorizer.vocabulary_)
v = vectorizer.transform(new)
print(v.toarray())

# convert to dataframe
q4 = pd.DataFrame(v.toarray())
# write the file to csv
q4.to_csv("q4_tfidf.csv", sep=',', float_format='%.15f')
# determine dimensions of vectors
q4.info() # 100 rows (representing 100 reviews) and 1384 columns (each item in the vector)

########## step 5 ##########

# initialize empty list to store POS tokens
POS = []
# POS tagging with for loop
for doc in reviews:
    token_doc = nltk.word_tokenize(doc)
    POS_token_doc = nltk.pos_tag(token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS.append(" ".join(POS_token_temp))

# set vectorizer to convert reviews to TDIDF with min doc freq per term = 4 and no n-gram specifications
vectorizer2 = TfidfVectorizer(min_df=4)
vectorizer2.fit(POS)
print(vectorizer2.vocabulary_)
POS_vector = vectorizer2.transform(POS)
print(POS_vector.toarray())

# convert to dataframe
q5 = pd.DataFrame(POS_vector.toarray())
# write the file to csv
q5.to_csv("q5_tfidf.csv", sep=',', float_format='%.15f')
# determine dimensions of vectors
q5.info() # 100 rows (representing 100 reviews) and 905 columns (each item in the vector)

########## part 2 ##########

from PIL import Image
import numpy as np
from pylab import *

# initialize empty list
images = []
hist_im = []
normalized_im = []
# image manipulation
for x in range(1,11):
    im = Image.open(str(x) + '.PNG')
    # resize image
    im_resize = im.resize((100,100))
    # change to greyscale
    im_grey_m = np.asarray(im_resize.convert('L'))
    # flatten the image
    im_v = im_grey_m.flatten()
    # add to list to write arrays into csv
    images.append(im_v)
    hist_im.append(im_v)
    imhist, bins = histogram(im_v, 256, normed = True)
    # change histogram to cdf curve
    cdf = imhist.cumsum()
    # multiply to convert/normalize image
    cdf = 255 * cdf/cdf[-1]
    im2 = interp(im_v, bins[:-1], cdf)
    normalized_im.append(im2)
print("images have been successfully converted and normalized")

# combine arrays for histogram
hist_im = np.concatenate(hist_im)
normalized_im = np.concatenate(normalized_im)

# step 3 histogram
plt.hist(hist_im.flatten(), 256)
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.title("Flattened Greyscale Histogram")

# step 4 histogram
plt.hist(normalized_im.flatten(), 256, color = 'orange')
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.title("Normalized Histogram")

# comparison curve
figure()
plt.hist(hist_im.flatten(), 256)
plt.hist(normalized_im.flatten(), 256)
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.title("Comparison Plot")

# convert to dataframe to save as csv
step2 = pd.DataFrame(images)
step2.to_csv("image_array.csv", sep = ',') # 10 rows and 10,000 columns
