# Twiiter-Sentimental-Analysis-with-online-streaming-data


                                              Manual

This project is based on the Sentimental Analysis of tweets(Adaptive and Predictive) in which we used tweets that is extracted
from twitter account using Twitter API and sentiment of each tweets is 
extracted from each tweets after going through data cleaning in the form of negative as -1 , positive as 1 and neutral as 0.
The tweets from API is in the form of JSON so to work on the data we have to use panda library of python 
to convert data in the form of data frame.

We have based this project for getting in depth of the tweets that the tweets made is positive ,neutral or negative.
and we can also predict the tweets based on the machine learning methodology.

We have used NLP library nltk for stopword and also textblob which is  a Python  library for processing textual data.
we also used CountVectorizer which is used in tokenization and ConllExtractor which is NounPhaseExtractor, WordNetLemmatizer()
which is used in Lemmatizing text which is used in getting data ready for data processing.
For Predictive Sentimental Analysis we have to train the data for this we have  used that same data extracted from the 
API and for that we have divided same data into train data set and test data set.

We have also used matplotlib for visualization 
piechart is used in tr6.py that give various sentiment based on the tweets and will give percentage of tweets as 
Positive
Weakly Positive
Strongly Positive
Negative
Weakly Negative
Strongly Negative

matplotlib is also used in visualiztion of confusion matrix defining
TP-- True Positive
FP-- False Positve
TN-- True Negative
FN-- False Negative

We have also used tweepy library for getting tweets ,We also need to Authenticiate our twitter developer account using
 ACCESS_TOKEN,ACCESS_SECRET,CONSUMER_KEY,CONSUMER_SECRET which is different for each user.For authenticiation we need to
authenticiate using OAuthHAndler library of tweepy

In order to execute this project do the following 
Install 
1  Python (https://www.python.org/downloads/release/python-374/)
Using command Prompt
2  pip(python get-pip.py)
3  textblob(pip install -U textblob)
4  numpy(pip install numpy)
5  matplotlib(pip install numpy)
6  sklearn(pip install scikit-learn)
7  tweepy(pip install tweepy)
8  pandas(pip install pandas)
9  nltk(pip install nltk)

For Adaptive sentimenal analysis of tweets that gives you pie chart based on the keyword/tag to be searched
Run the following commands----
python tr6.py

Then enter the keyword and no. of tweets you want to analyze

Note--If we want the sentimental analysis based on the particular user 
Run the following command----
python tr2.py

For Predictive Sentimental Analysis of tweets that implies various machine learning algorithms and gives the accuracy
of each of the algorithms with the confusion matrix based on the keyword/tag to be searched
Run the following commands ----
python tr7.py 

Then enter the keyword and no. of tweets you want to analyze.
We will get first confusion matrix 

Different methodology used in machine learning are
AdaBoost
LogisticRegression
KNeighborsClassifier
SVC(type of SVM classifier)
RandomForestClassifier
GaussianNB()

