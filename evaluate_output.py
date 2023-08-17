#import tuning_parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical


class reg_model_evaluation():
    
    def evaluate_reg_model(self, y_test, y_test_pred):
        MAE = mean_absolute_error(y_test,y_test_pred)
        RMSE = mean_squared_error(y_test,y_test_pred)**0.5
        return MAE, RMSE 

    def plot_residual_error(self, y_test, y_test_pred):
        # reshape y_pred to have same shape with y. 
        #print(y_test.shape) #(402,)
        #print(y_test_pred.shape)#(402,1)
        y_test_pred_reshaped = np.reshape(y_test_pred, y_test.shape)
        residual_err = y_test - y_test_pred_reshaped
        sns.scatterplot(x=y_test, y=residual_err)
        plt.xlabel("y_test")
        plt.ylabel("Residual")
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()


    def plot_predictions(self, y_test, y_test_pred):
        y_test_pred_reshaped = np.reshape(y_test_pred, y_test.shape)
        plt.scatter(x=y_test, y=y_test_pred_reshaped)
        plt.plot(y_test, y_test, 'r')
        plt.xlabel("y")
        plt.ylabel("prediction_y_reshaped")
        plt.show()

class cate_model_validation():

    def plot_losses(self, losses):
        losses.plot()
    
    # plot_classification_report only takes 1d. 
    def plot_classification_report(self, y_test, y_test_pred):
        print(y_test.shape)#(45, 3)
        print(y_test_pred.shape)#(45, 3)
        
        #print (f"y_test_pred after getting dummies is {y_test_pred}")
        y_test_pred = np.argmax(y_test_pred, axis=1) # 1d
        y_test = np.argmax(y_test, axis=1) 
        print(f"y_test is {y_test}")
       

        """
        print(f"y_test_pred after argmax is {y_test_pred}")
        y_test_pred = to_categorical(y_test_pred, num_classes = np.max(y_test)+1) # numclass is outputsize 
        print(f"y_test_pred after categotical() is {y_test_pred}")
        """
        print(classification_report(y_test, y_test_pred)) 
        print ("test more time and it gives different accuracy")

        
