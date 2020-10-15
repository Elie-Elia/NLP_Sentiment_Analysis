import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
import string
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from nltk.classify import NaiveBayesClassifier
from sklearn import preprocessing
import sklearn.metrics
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

ISO_ENCODING = "ISO-8859-1"
positiveLinesFP = './rt-polaritydata/rt-polarity.pos'
negativeLinesFP = './rt-polaritydata/rt-polarity.neg'
stop_words = set(stopwords.words('english')) 
negativeLinesList = []
positiveLinesList = []
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

def accept_data_input(): 

    f = open(positiveLinesFP,"r", encoding= ISO_ENCODING)
    positiveLinesList = [line.rstrip('\n') for line in f]

    f = open(negativeLinesFP,"r", encoding= ISO_ENCODING)
    negativeLinesList = [line.rstrip('\n') for line in f]

    y = [1 for line in positiveLinesList] + [0 for line in negativeLinesList]
    x = positiveLinesList + negativeLinesList
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

    return ([xTrain, yTrain], [xTest, yTest])
    
def basic_preprocessing(text):
    
    
    text = re.sub('\\W', " ", text)
    
    word_tokens = word_tokenize(text)

    filtered_sentences = [] 
      
    for w in word_tokens: 
        if w not in stop_words and w not in string.punctuation:
            filtered_sentences.append(w) 
 
    return filtered_sentences

def lemmatizer_preprocessor(text):
    
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    basic_filtering_list = basic_preprocessing(text)
    lemmatized_words_list = []
    
    for token, tag in pos_tag(basic_filtering_list):
        lemmatized_words_list.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))
    
    
    return ' '.join(lemmatized_words_list)
    
    
def stemmer_preprocessor(text) :

    basic_filtering_list = basic_preprocessing(text)
    
    stemmed_words_list = ' '.join([stemmer.stem(w) for w in basic_filtering_list])
    
    return stemmed_words_list
    
    
def train_preprocessing(xTrain, yTrain):
    
    #Change preprocessor to stemmer when wanting to run tests for stemming
    count_vector = CountVectorizer(min_df=1, preprocessor= lemmatizer_preprocessor)
    xTrain_count_vector= count_vector.fit_transform(xTrain)
    
    feature_vector = xTrain_count_vector.toarray()
    feature_vector_names = count_vector.get_feature_names()
    
    return feature_vector, feature_vector_names, count_vector
    
def test_preprocessing (xTest, count_vector) :
    
    xTest_count_vector = count_vector.transform(xTest)
    feature_vector = xTest_count_vector.toarray()
    return feature_vector
    
def LR_classifier(feature_vector_train, feature_vector_test, yTrain, yTest):

    modelLR = LogisticRegression(random_state=0, C=2, max_iter=10000)
    modelLR.fit(feature_vector_train, yTrain)
    predicted_results = modelLR.predict (feature_vector_test)
    return accuracy_score(yTest, predicted_results)
    
def SVM_classifier (feature_vector_train, yTrain, feature_vector_test, yTest):
    
    modelSVM = svm.LinearSVC()
    modelSVM.fit(feature_vector_train, yTrain)
    predicted_results = modelSVM.predict(feature_vector_test)
    return accuracy_score(yTest, predicted_results)
    
def random_baseline_classifier (feature_vector_test, yTest):
    
    predictions=[]
    
    for i in range(len(feature_vector_test)):
        predictions.append(random.randint(0,1))
        
    return accuracy_score(yTest, predictions)

def SGDC_Classifier (feature_vector_train, feature_vector_test, yTrain, yTest):
    
    modelSGDC = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000)
    modelSGDC.fit(feature_vector_train, yTrain)
    predicted_results = modelSGDC.predict(feature_vector_test)
    return accuracy_score(yTest, predicted_results)

def NB_Classifier (feature_vector_train, feature_vector_test, feature_vector_names, yTrain, yTest):
     
     
    NB_dataset=[]
    for idx in range (len(feature_vector_train)):
        label = yTrain[idx] #positive or negative
        map = {}
      
        for col in range (len(feature_vector_names)):
            word = feature_vector_names[col]
            freq = feature_vector_train[idx][col]
            map[word] = freq
        NB_dataset.append( (map, label) )
      
    modelNB = NaiveBayesClassifier.train(NB_dataset)
 
    
    testdataset = []
    for idx in range (len(feature_vector_test)):
        map = {}

        for col in range (len(feature_vector_names)):
            word = feature_vector_names[col]
            freq = feature_vector_test[idx][col]
            map[word] = freq
            

        testdataset.append( map )
     
    predicted_results = modelNB.classify_many(testdataset)
    test_confusion_matrix = sklearn.metrics.confusion_matrix(yTest, predicted_results)
    print(test_confusion_matrix)
    return accuracy_score(yTest, predicted_results)

data_to_train, data_to_test = accept_data_input()

xTrain, yTrain = data_to_train
xTest, yTest = data_to_test

feature_vector_train, feature_vector_names, count_vector = train_preprocessing(xTrain, yTrain)
feature_vector_test = test_preprocessing(xTest,count_vector)


LR_classifier_accuracy = LR_classifier(feature_vector_train, feature_vector_test, yTrain, yTest) 
print ("Linear Regression Accuracy:" ,LR_classifier_accuracy)

random_baseline_accuracy = random_baseline_classifier(feature_vector_test, yTest)
print("Random Baseline Accuracy:" ,random_baseline_accuracy)

SVM_classifier_accuracy = SVM_classifier(feature_vector_train, yTrain, feature_vector_test, yTest)
print("SVM Classifier Accuracy:" , SVM_classifier_accuracy)

#Stochastic Gradient Descent loss="hinge": (soft-margin) linear Support Vector Machine
SGDC_Classifier_accuracy = SGDC_Classifier(feature_vector_train, feature_vector_test, yTrain, yTest)
print("Stochastic Gradient Descent Accuracy:" ,SGDC_Classifier_accuracy)

NB_Classifier_accuracy = NB_Classifier(feature_vector_train, feature_vector_test, feature_vector_names, yTrain, yTest)
print("Naive Bayes Classifier Accuracy:",NB_Classifier_accuracy)