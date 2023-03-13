import nltk  # NLP toolbox
from os import getcwd
import pandas as pd  # Library for Dataframes
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt  # Library for visualization
import numpy as np  # Library for math functions
import re  # library for regular expression operations
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets  ## Concatenate the lists.
labels = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)

# split the data into two pieces, one for training and one for testing (validation set)
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg

print("Number of tweets: ", len(train_x))

# "C:\Users\jamme\Documents\logistic_features.csv"
data = pd.read_csv("C:/Users/jamme/Documents/logistic_features.csv")  # Load a 3 columns csv file using pandas function
data.head(10)  # Print the first 10 data entries

# get rid of the data frame to keep only Numpy arrays.
# Each feature is labeled as bias, positive and negative
X = data[['bias', 'positive', 'negative']].values  # Get only the numerical values of the dataframe
Y = data['sentiment'].values  # Put in Y the corresponding labels or sentiments

print(X.shape)  # Print the shape of the X part
print(X)  # Print some rows of X

# Load a pretrained Logistic Regression model
#  a Logistic regression model must be trained. The next cell contains the resulting model from such training.
#  Notice that a list of 3 numeric values represents the whole model, that we have called theta  𝜃

theta = [6.03518871e-08, 5.38184972e-04, -5.58300168e-04]

# Plot the samples in a scatter plot
# here we ignore the bias for clearer presentation. But it should be a 3 D feature space.

# Plot the samples using columns 1 and 2 of the matrix
fig, ax = plt.subplots(figsize=(8, 8))

colors = ['red', 'green']

# Color based on the sentiment Y
ax.scatter(X[:, 1], X[:, 2], c=[colors[int(k)] for k in Y], s=0.1)  # Plot a dot for each pair of words
plt.xlabel("Positive")
plt.ylabel("Negative")


#  Plot the model alongside the data
# draw a gray line to show the cutoff between the positive and negative regions + direction of the line

# Equation for the separation plane
# It give a value in the negative axe as a function of a positive value
# f(pos, neg, W) = w0 + w1 * pos + w2 * neg = 0
# s(pos, W) = (-w0 - w1 * pos) / w2
def neg(theta, pos):
    return (-theta[0] - pos * theta[1]) / theta[2]


# Equation for the direction of the sentiments change
# We don't care about the magnitude of the change. We are only interested
# in the direction. So this direction is just a perpendicular function to the
# separation plane
# df(pos, W) = pos * w2 / w1
def direction(theta, pos):
    return pos * theta[2] / theta[1]


# The green line in the chart points in the direction where z > 0 and the red line points in the direction where z < 0.
# The direction of these lines are given by the weights  𝜃1 and  𝜃2
# Plot the samples using columns 1 and 2 of the matrix
fig, ax = plt.subplots(figsize=(8, 8))

colors = ['red', 'green']

# Color base on the sentiment Y
ax.scatter(X[:, 1], X[:, 2], c=[colors[int(k)] for k in Y], s=0.1)  # Plot a dot for each pair of words
plt.xlabel("Positive")
plt.ylabel("Negative")

# Now lets represent the logistic regression model in this chart.
maxpos = np.max(X[:, 1])

offset = 5000  # The pos value for the direction vectors origin

# Plot a gray line that divides the 2 areas.
ax.plot([0, maxpos], [neg(theta, 0), neg(theta, maxpos)], color='gray')

# Plot a green line pointing to the positive direction
ax.arrow(offset, neg(theta, offset), offset, direction(theta, offset), head_width=500, head_length=500, fc='g', ec='g')
# Plot a red line pointing to the negative direction
ax.arrow(offset, neg(theta, offset), -offset, -direction(theta, offset), head_width=500, head_length=500, fc='r',
         ec='r')

plt.show()
