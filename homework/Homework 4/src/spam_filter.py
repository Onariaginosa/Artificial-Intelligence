'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text 
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.
        
        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """
        # [!] TODO
        self.vectorizer = CountVectorizer(input = 'content',stop_words = 'english')
        self.features =self.vectorizer.fit_transform(text_train)
        #print(self.features)
        self.clf = MultinomialNB()
        self.clf.fit(self.features,labels_train)
        #print(self.test)

        
        
    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text
        
        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.
        """
        # [!] TODO

        test = self.vectorizer.transform(text_test)
        result = self.clf.predict(test)
        return result
    
    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        """
        # [!] TODO
        result = self.classify(text_test)
        print(classification_report(labels_test,result))
        
    
        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels
    """
    # [!] TODO
    data = pd.read_csv(data_file, encoding = "latin-1")
    #print(data.head())
    #print(data.columns)
    #data = data.drop("v1",axis=1)
    data = data.drop(data.iloc[:,2:data.shape[1]],axis=1)
    data = data.rename(columns = {"v1" : "class", "v2": "text" })
    #print(data.head())
    return data


if __name__ == "__main__":
    # [!] TODO
    data = load_and_sanitize("../dat/texts.csv")
    X_train, X_test, y_train, y_test = train_test_split(data["text"],data["class"])
    spamfil = SpamFilter(X_train,y_train)
    print(spamfil.classify(["Hi ,A user just logged into your Facebook account from a new device Samsung Galaxy S10. We are sending you this email to verify it's really you. "]))
    print(spamfil.classify(["This is to notify you that Series of meetings have been held over the past (1) Month now with the Secretary General of the United Nations Organization United State of America, which ended Today been Friday Dated 27th of November 2020, It is obvious that you have not received your fund which is now in the amount of ($10Million) as a compensation award to you, due"]))
    print(spamfil.classify(["Hey Tommy, Can you send over last weeks report? I want to review some of details"]))
    print(spamfil.classify(["Hi this is Mom, Merry Christmas. I miss you, I love you"]))
    print(spamfil.test_model(X_test,y_test))


    #print(data["class"])

    