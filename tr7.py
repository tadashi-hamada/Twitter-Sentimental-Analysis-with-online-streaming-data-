from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from tr4 import *


textblob_key_phrases = []
extractor = ConllExtractor()

for index, row in tweets_df.iterrows():
    # filerting out all the hashtags
    words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]
    
    hash_removed_sentence = ' '.join(words_without_hash)
    
    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)
    textblob_key_phrases.append(list(blob.noun_phrases))

#print(textblob_key_phrases[:10])  


tweets_df['key_phrases'] = textblob_key_phrases
#print(tweets_df.head(10))         
tweets_df2 = tweets_df[tweets_df['key_phrases'].str.len()>0]

target_variable = tweets_df2['sentiment'].apply(lambda x: 0 if x==-1 else 1)
bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')
# bag-of-words feature matrix
bow_word_feature = bow_word_vectorizer.fit_transform(tweets_df2['absolute_tidy_tweets'])


phrase_sents = tweets_df2['key_phrases'].apply(lambda x: ' '.join(x))

# BOW phrase features
bow_phrase_vectorizer = CountVectorizer(max_df=0.90, min_df=2)
bow_phrase_feature = bow_phrase_vectorizer.fit_transform(phrase_sents)



def plot_confusion_matrix(matrix):
    plt.clf()
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Set2_r)
    classNames = ['Positive', 'Negative']
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TP','FP'], ['FN', 'TN']]

    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(matrix[i][j]))
    plt.show()

def naive_model(X_train, X_test, y_train, y_test):
    naive_classifier = GaussianNB()
    naive_classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = naive_classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for GaussianNB - {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

    classifier = AdaBoostClassifier()
    classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for AdaBoostClassifier- {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

    classifier =RandomForestClassifier(n_estimators=200)
    classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for RandomForestClassifier- {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

    

    classifier = SVC(kernel="rbf", C=0.025, probability=True,gamma='auto')
    classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for SVC- {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

    classifier = KNeighborsClassifier(3)
    classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for KNeighborsClassifier- {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)

    classifier = LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200)
    classifier.fit(X_train.toarray(), y_train)

    # predictions over test set
    predictions = classifier.predict(X_test.toarray())

    # calculating Accuracy Score
    print(f'Accuracy Score for LogisticRegression- {accuracy_score(y_test, predictions)}')
    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    plot_confusion_matrix(conf_matrix)


X_train, X_test, y_train, y_test = train_test_split(bow_word_feature, target_variable, test_size=0.3, random_state=272)
naive_model(X_train, X_test, y_train, y_test)

