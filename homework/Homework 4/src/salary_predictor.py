'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

class SalaryPredictor:

    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.
        
        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        # [!] TODO
        ####
        #defined columns
        columns_to_encode=["work_class","education","marital","occupation_code","relationship","race","sex","country"]
        #columns_to_scale=["age","education_years","capital_gain","capital_loss","hours_per_week"]
        columns_to_scale=["age","education_years","hours_per_week"]
        columns_to_leave= ["capital_gain","capital_loss"]
        self.columns_with_missing=["work_class","occupation_code","country"]
        #get the categories for each column
        cat = []
        for col in X_train.columns:
            if pd.api.types.is_string_dtype(X_train[col]):
                cat.append(X_train[col].unique())

        #print(cat)
        self.imp = SimpleImputer(missing_values = "?", strategy="most_frequent")
        imp_columns = self.imp.fit_transform(X_train[columns_to_encode])
        
        self.ohe = preprocessing.OneHotEncoder(  handle_unknown = "ignore",categories = cat,sparse = False)
        self.le2 = preprocessing.LabelEncoder()
        self.Kbin = preprocessing.KBinsDiscretizer(n_bins=[2,2,2],encode="ordinal")
        self.scale = preprocessing.StandardScaler()
        #scaled_columns = self.scale.fit_transform(X_train[columns_to_scale])
        #ds_columns = self.Kbin.fit_transform(scaled_columns)
        scaled_columns = self.scale.fit_transform(X_train[columns_to_leave])
        ds_columns = self.Kbin.fit_transform(X_train[columns_to_scale])
        #print(X_train)
        encoded_columns = self.ohe.fit_transform(imp_columns)
        #print(X_train)
        #print(encoded_columns)
        #for col in encoded_columns:
            #print(col)
        #   if col == "work_class" or col == "occupation_code" or col == "country":
        #        encoded_columns[col].replace([0],-1)
        processed_data = np.concatenate([ds_columns,encoded_columns,scaled_columns],axis=1)
        #self.imp = SimpleImputer(missing_values = -1, strategy="most_frequent")
        #X_train = self.imp.fit_transform(processed_data)
        self.le = preprocessing.LabelEncoder()
        #y_process = self.le.fit_transform(y_train)
        #print(y_process)
        self.clf = LogisticRegression(max_iter=1000).fit(processed_data,y_train)

       
    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!
        
        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        # [!] TODO
        columns_to_encode=["work_class","education","marital","occupation_code","relationship","race","sex","country"]
        #columns_to_scale=["age","education_years","capital_gain","capital_loss","hours_per_week"]
        columns_to_scale=["age","education_years","hours_per_week"]
        columns_to_leave= ["capital_gain","capital_loss"]
        ds_columns = self.Kbin.transform(X_test[columns_to_scale])
        imp_columns= self.imp.transform(X_test[columns_to_encode])
        scaled_columns = self.scale.transform(X_test[columns_to_leave])
        encoded_columns = self.ohe.transform(imp_columns)
        processed_data = np.concatenate([ds_columns,encoded_columns,scaled_columns],axis=1)
        result = self.clf.predict(processed_data)
        return result
    
    def test_model (self, X_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.
        
        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        # [!] TODO
        result = self.classify(X_test)
        #y_test = self.le.transform(y_test)
        print(classification_report(y_test,result))
        

        
def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.
    
    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    # [!] TODO
    data = pd.read_csv( data_file, encoding = "latin-1", skipinitialspace = True)
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].str.strip()
            #source : https://towardsdatascience.com/dealing-with-extra-white-spaces-while-reading-csv-in-pandas-67b0c2b71e6a
    #for col in data.columns:
        #if pd.api.types.is_string_dtype(data[col]):
            #unique = data[col].unique()
            #print("Unique data for col: ", col, " is ", unique)
    return data


if __name__ == "__main__":
    # [!] TODO
    data = load_and_sanitize("../dat/salary.csv")
    #print(data[data.columns[len(data.columns)-1]])
    #print(data[data.columns[0:len(data.columns)-1]])
    X_train, X_test, y_train, y_test = train_test_split(data[data.columns[0:len(data.columns)-1]],data[data.columns[len(data.columns)-1]])
    test = SalaryPredictor(X_train,y_train)
    print(test.test_model(X_test,y_test))
    #print(test.classify(X_test))
    
    