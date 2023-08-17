import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MyANN_classification():

    early_stop = EarlyStopping(monitor = 'val_loss', patience = 5,\
                                    mode ="min", verbose =1)
    def build_binary_class_model(self,input_size, scaled_X_train, y_train, scaled_X_test, y_test):

        # if features are less than 30, fist hidden layer 1/2 and int round. 
        if input_size >30:
            hidden_layer_size = (input_size, input_size//2, 0.25, input_size//4,1 )
        else:
            hidden_layer_size = (input_size, input_size, 0.00, input_size//4,1)
        # build network
        model = Sequential()
        # input 
        model.add(Dense(input_size, activation = "relu"))
        #hidden layers
        model.add(Dropout(hidden_layer_size[2]))
        model.add(Dense(hidden_layer_size[0], activation = "relu")) 
        model.add(Dropout(hidden_layer_size[2]))            
        model.add(Dense(units = hidden_layer_size[1], activation = "relu"))
        model.add(Dense(units = hidden_layer_size[3], activation = "relu"))
        model.add(Dense(units =1, activation = "sigmoid"))

        model.compile(loss = "binary_crossentropy", optimizer = "adam")

        model.fit(scaled_X_train, y_train,\
                   epochs = 600, \
                    validation_data = (scaled_X_test, y_test), 
                        verbose =1, callbacks = [self.early_stop])
        
        losses = pd.DataFrame(model.history.history)
        # recall: true positives out of all actual positive \
        # (when want to dentify as many positive cases as possible,) 
        # F1 score mean of precision and recall, 
        y_test_pred = model.predict(scaled_X_test)
        y_test_pred = (y_test_pred>0.5).astype(int)
        # it seems that f1 and recall does not show correctly as classification_report
        recalled_score = recall_score(y_test, y_test_pred)
        f1_scr = f1_score(y_test, y_test_pred)
        print(f"model_loss is {losses}, recall_score is{recalled_score}\
              ,F1 score is {f1_scr}")
        return model, losses, y_test_pred


    def build_multi_class_model(self,input_size, scaled_X_train,\
                                 y_train, scaled_X_test, y_test, output_size):
        
        if input_size >30:
            hidden_layer_size = (input_size, input_size//2, 0.25, input_size//4, output_size)
        else:
            hidden_layer_size = (input_size, input_size, 0.00, input_size//4,output_size)

        #build model:
        model = Sequential()

        #input layer
        model.add(Dense(input_size, activation = "relu"))

        #hidden layer
        model.add(Dropout(hidden_layer_size[2]))
        model.add(Dense(hidden_layer_size[0], activation = "relu"))
        model.add(Dense(hidden_layer_size[0], activation = "relu"))
        model.add(Dropout(hidden_layer_size[2]))
        model.add(Dense(units = hidden_layer_size[1], activation = "relu"))
        model.add(Dense(units = hidden_layer_size[3], activation = "relu"))

        # output layer
        model.add(Dense(units = output_size , activation = "softmax"))

        # Compile
        model.compile(optimizer = "adam", loss = "categorical_crossentropy"\
                      , metrics = ["accuracy"])

        # fit the model
        model.fit(scaled_X_train, y_train, 
          epochs=600, 
          validation_data=(scaled_X_test, y_test), 
          verbose=1, callbacks = [self.early_stop])

        # predict 
        losses = pd.DataFrame(model.history.history)        
        predictions = model.predict(scaled_X_test)
        # prediction has only 2 
        #y_test_pred_multi = pd.get_dummies(predictions)    
        return model, losses, predictions