import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# this class makes data suitable for model.
class Data_preparation():

    # this function checks if data has missing value, and convert object features.
    def validate_df(self,df, label):
        X =df.drop(label, axis =1)
        cols = X.select_dtypes(include = "object")
        # If data has missing value, raise ValueError.
        if  df.isnull().sum().any():
            #If data has missing values, raise error. 
            raise ValueError("data has missing values, you need to fill up and rerun")
        # if features has object data type, it will be converted to numeric. 
        elif not cols.empty: 
            df_features = pd.get_dummies(data=X)
            new_features = pd.DataFrame(df_features)
            input_size = new_features.shape[1]
            print(f"input size here is {input_size}")
            label_df = pd.DataFrame(df[label])
            valid_df = pd.concat([new_features, label_df], axis=1)
            # valid_df has all features in numeric datatype.
            # X = new_features.values, X for all models
            return valid_df, input_size, new_features.values
        else:
            input_size = X.shape[1]
            print(f"input size here is {input_size}")
            return df, input_size, X
    
    def decide_label(self, valid_df, label):
        columns = valid_df.columns
        if label in columns:
            label_exist = valid_df[label]# Series
            return label_exist
        raise ValueError("Label is not in your dataframe")

    # This function checks if data fits regression models.    
    def get_label_values_for_regression(self, label_exist):
        label_values = label_exist.values
        print(label_values)
        enough_unique = len(np.unique(label_exist))>10
        # If label values are numeric, return valid data and label for regression models.
        if is_numeric_dtype(label_values) and enough_unique:
            # y for regression model is label_values 
            # label_exist is series, so .name
            #X = valid_df.drop(label_exist.name, axis=1).values
            #print (X, y)
            return label_values
        raise ValueError("data is not suitable for regression models.") 
    
    # This function validate if data fits classification models.
    def  get_label_values_for_classifiers(self, label_exist):
        # label_exist is pd Series
        unique_values = np.unique(label_exist) #numpy.ndarray
        #print(unique_values)
        length_is_small = len(unique_values) <= 10
        all_values_are_int = np.issubdtype(unique_values.dtype, np.integer)
        valid_values = set(unique_values) <= set(range(10))

        # if label is a str column or int for category, it fits classification models. 
        if is_string_dtype(unique_values):
            if len(unique_values) == 2:
                valid_label = pd.get_dummies(data =label_exist, drop_first=True)
            if len(unique_values)>2:
                valid_label = pd.get_dummies(data =label_exist)
            # y is valid_label
            return valid_label.values 
        elif length_is_small and all_values_are_int and valid_values:
            # y is label_exist
            return label_exist.values
        else:
            raise ValueError("label is not suitable for classification models")
        
    def split_data(self, X, y):
        #print("in prprocess, argument X, y")
        #print(X, y)
       
        if X.shape[0] < 1000:
            test_size = 0.3
            print("test size is 30%")
        else:
            test_size = 0.2
            print("test size is 20%")

        X_train, X_test, y_train, y_test = train_test_split\
                                    (X, y, test_size=test_size, random_state=101)
        #print(f" i preprocess if X and y no problem, X_train is {X_train}, X_test is {X_test}, y_test is {y_test}")
            #print(X_train.shape, y_train.shape, X_test.shape. y_test.shape)
        return X_train, X_test, y_train, y_test
        

    def scale_features(self, X_train, X_test):
        #print(f" trian and test are from split, so if the X, y no prob\
              #xtrain and xtest should not be prob. \
              #argument X_train is {X_train}, X_test is {X_test}")
        scaler = MinMaxScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        #print(type(scaled_X_test))
        return scaled_X_train, scaled_X_test
# the quotation mark
