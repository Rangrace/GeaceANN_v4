from typing import List, Union
import pandas as pd
import validate_path
import preprocess
import evaluate_output
import numpy as np
from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping




class MyAnn ():

    """ 
        MyAnn is a class to build multilayer Perceptron classifier 
        or regressor by reading in and validation of paramters.
        parameters:

            data_set : takes absolut data path, we assume that the data is valid if the path 
                is valid, for example it should not have missing values or string type features. 
            target: take in str argument.
            hidden_layer_size : it is a tuple that indicates input size, hidden layer size,
              dropout rate and output size.
            activation: calculation logic for all the layers after input layer.
                choose from ("relu", "sigmoid", "softmax", "tanh").
            optimizer: method that finds the minimum loss for the model
                choose from ('adam', 'rmsprop', 'sgd').
            batch_size: how many rows sending in the training model every round. 
                    batch_size = 2**n 
            epochs: how many time should the model be trained. 
            monitor: mornitor model performance based on values that is assigned. 
                for example mornitor = "accuracy", accuracy metric will be monitored
                during training. 
                values choose from ("val_loss" , "accuracy")
            patience : patience for early stop if asigned will be number of epochs
                (int)
            mode : The mode parameter is typically used in combination with other 
                    parameters, such as monitor and patience, 
                    to determine the stopping criterion during training.
                    values choose from ('auto', 'min', 'max')
            verbose : how much information shown during training the model
                        values choose form range(0,1,2)
            use_multiprocessing : use multiprocessing or single process when loading data. 
                when setting it to "True" it is beneficial for multi-core CPU. 
                        (False, True)

    """
    # class attributes:
    classes_: Union[int, None] = None   
    loss_: Union[str, None] = None   
    best_loss: Union[str, None] = None  
    features_: Union[str, List[str], None] = None    
    n_layers: Union[int, None] = None   
    n_outputs_: Union[int, None] = None
    out_activation_: Union[str, None] = None
    best_monitor_: Union[str, None] = None

    def __init__(self, data_set:str, target:str, 
                hidden_layer_size:tuple = (100,), activation:str = "relu",
                loss:str = 'mse', optimizer:str = 'adam', batch_size:int = 32,
                epochs:int = 1, monitor:str = 'val_loss', patience:int = None,
                mode:str = 'auto', verbose:int = 1,
                use_multiprocessing:bool = False):
        self.data_set = data_set
        self.target = target

        if isinstance(hidden_layer_size, (list, tuple)):
            self.hidden_layer_size = hidden_layer_size
        else:
            raise ValueError("hidden_layer_size must be a tuple or list.")
        
        if activation in ["relu", "sigmoid", "softmax", "tanh"]:
            # can be ussed for all reg, binary and multiclass models. 
            self.activation = activation
        else:
            raise ValueError("input must be one of ['relu', 'sigmoid', 'softmax','tanh']")
        
        if loss in ['mse', 'binary_crossentropy', 
                        'categorical_crossentropy']:
            # suit for different types of models respectively.
            self.loss = loss
        else:
            raise ValueError("input must be one of \
                             ['mse', 'binary_crossentropy','categorical_crossentropy']")
        
        if optimizer in ['adam', 'rmsprop', 'sgd']: 
            # all these 3 can be used for all types of models. reg, bin and multi-
            self.optimizer = optimizer
        else:
            raise ValueError("input must be one of \
                             ['adam', 'rmsprop', 'sgd']")
        
        if batch_size in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
            self.batch_size = batch_size
        else:
            raise ValueError(batch_size, 'is not an 2**(0, 13) intiger')
        
        if isinstance(epochs, int) and epochs > 0:
            self.epochs = epochs
        else:
            raise ValueError('epochs must be a positive intiger')

        if monitor in ['val_loss', 'accuracy']:
            # val_loss for reg, accuracy for bin and multi
            self.monitor = monitor
        else:
            raise ValueError("input must be one of \
                             ['val_loss', 'accuracy']")   
         
        if mode in ['auto', 'min', 'max']:
            # min for loss, max for matric, auto for auto-adapting.  
            self.mode = mode 
        else:
            raise ValueError("input must be one of ['auto', 'min', 'max']") 
       
        if verbose in range(0,3):
            self.verbose = verbose
        else:
            raise ValueError("input can only be 0, 1 or 2")
        
        if use_multiprocessing in [False, True]:
            self.use_multiprocessing = use_multiprocessing
        else:
            raise ValueError("input can only be False or True.")
        
        if isinstance(patience, int) and patience>0:
            self.patience = patience
        else:
            raise ValueError("Patience needs to be a positive intiger")
        
        self.n_layers_ = len(self.hidden_layer_size) + 2

    # It sets an instance variable. Rename and redocument function.
    """ 
    this function checks if data path is valid.
    """    
    def validate_path(self):
        
        self.df = validate_path.Data_collection().validate_path(self.data_set)
    
    
    """ 
    this function will valid data set to see if it is ready for trainging models. 
    It assigns valid df, input size and X to self.

    This function also fills dummies for "o" type features and check missing 
    values for whole df. 
    """
    def valid_df(self): 
        valid_df, input_size, X = preprocess.Data_preparation().validate_df\
                                                                (self.df, self.target)
        self.features_ = X.columns
        self.input_size = input_size
        self.X = X.values
        self.validate_df = valid_df
        # self.target is str, df[] becomes series, but did not return series, instead, error... 

        self.label = preprocess.Data_preparation().decide_label(valid_df, self.target)
    
    """
    This function chooses different model parameters based on target type. 
    """
    def choose_algarithm(self):
        
        # choose algarithm based on target type
        target_type = type_of_target(self.label.values)

        # O: Good with if statement and sub functions. Switch / case is more suitable though.
        if  target_type == "continuous":
            self._tune_params_for_regression()

        elif target_type == "binary":
            self._tune_params_for_binary()

        elif target_type == "multiclass":
            self._tune_params_for_multiclass()

        else:
            raise ValueError("Your target does not suit continuous or binary or multiclass models.")


    def _tune_params_for_regression(self):

        self.y = self.label.values
        self.n_outputs_ = 1
        #self.classes_ = self.y # series or self.classes_ = self.target -> str?
        self.classes_ = None
        self.best_loss = "mse"
        if self.mode == "max":
            raise ValueError("for regresson models, mode should be set either to auto\
                                or min")
        self.best_monitor = "val_loss"
        if self.monitor != self.best_monitor:
            raise ValueError(f"{self.best_monitor}is the best monitor for \
                             your type of data.")
        self.out_activation_ = None

    
    def _tune_params_for_binary(self):

        self.dummy_label = pd.get_dummies(data =self.label, drop_first=True)
        self.y = self.dummy_label.values
        self.n_outputs_ = 1
        self.classes_ = np.unique(self.label) # returns array with unique values. 
        self.best_loss = "binary_crossentropy"
        if self.mode == "min":
            raise ValueError("for clssification models, mode should be set either to auto\
                                or max")
        self.best_monitor = "accuracy"
        if self.monitor != self.best_monitor:
            raise ValueError(f"{self.best_monitor}is the best monitor for \
                             your type of data.")
        self.out_activation_ = "sigmoid"

    def _tune_params_for_multiclass(self):
        
        self.dummy_label = pd.get_dummies(data = self.label)
        self.y = self.dummy_label.values
        self.n_outputs_ = len(np.unique(self.label))
        self.classes_ = np.unique(self.label)
        self.best_loss = "categorical_crossentropy"
        if self.mode == "min":
            raise ValueError("for clssification models, mode should be set either to auto\
                                or max")
        self.best_monitor = "accuracy"
        if self.monitor != self.best_monitor:
            raise ValueError(f"{self.best_monitor}is the best monitor for your\
                              type of data.")
        self.out_activation_ = "softmax"

    #split
    def split(self):

        self.X_train, self.X_test, self.y_train, self.y_test = \
        preprocess.Data_preparation().split_data(self.X, self.y)

       #scale
    def scale_data(self):

        self.scaled_X_train,self.scaled_X_test = \
        preprocess.Data_preparation().scale_features(self.X_train, self.X_test)
       
    # Separate into sub functions. Maybe 'add_models', 'compile_models' and 'fit_model'
    def build_models(self):

        self.model = Sequential()

        #input layer
        self.model.add(Dense(units = self.input_size, activation = self.activation))

        #hidden layer
        # we expect users put in suitable values in hidden layer. 
        for layer in self.hidden_layer_size:
            if layer > 0 and type(layer) == int:
                self.model.add(Dense(layer, activation = self.activation))
            
            elif layer <0 and layer >-1:
                self.model.add(Dropout(abs(layer)))
            
            elif layer == 0:
                self.model.add(Dropout(abs(layer)))
            else:
                raise Exception(" value in hidden layer must be postive \
                                int or negtive float between 0 and -1")
                
        #output layer
        self.model.add(Dense(self.n_outputs_, activation = self.out_activation_))

        #compile
        if self.loss != self.best_loss:
            raise ValueError(f'{self.best_loss} is more suitable loss for your data.') 
        
        #classification compling
        if self.out_activation_:
            self.model.compile(optimizer=self.optimizer, loss = self.loss, metrics = ['accuracy'])

        #regression compling
        else:
            self.model.compile(optimizer=self.optimizer , loss = self.loss)

        #train the model with early stoping.
        early_stop = EarlyStopping(monitor=self.monitor,
            mode = self.mode, patience=self.patience, verbose=self.verbose)

        self.model.fit(x=self.scaled_X_train,
            y=self.y_train,
            validation_data = (self.scaled_X_test, self.y_test),
            epochs = self.epochs,
            callbacks = [early_stop],
            verbose=self.verbose,
            batch_size = self.batch_size,
            use_multiprocessing = self.use_multiprocessing)
        
        self.model_loss = pd.DataFrame(self.model.history.history)
        self.model.summary()

    def predict(self):

        self.y_pred = self.model.predict(self.scaled_X_test)
        print (f"y_pred shape is {self.y_pred.shape}") 
        return self.y_pred
    
    def evaluate_output(self):
        # O: don't call predict here. Use 'y_pred' as a function input argument instead.
        #y_pred = self.predict() # when calling it no need of self? O: You need self since
        # predict is an instance function.

        # how to plot mae
        if self.loss == "mse":
            reg_model_evalutation_object= evaluate_output.reg_model_evaluation()
            reg_model_evalutation_object.plot_residual_error\
                (self.y_test, self.y_pred)
            reg_model_evalutation_object.plot_predictions(self.y_test, self.y_pred)
        else:
            cate_model_validation_object = evaluate_output.cate_model_validation()
            cate_model_validation_object.plot_losses(self.model_loss)  
            cate_model_validation_object.\
                plot_classification_report(self.y_test, self.y_pred)
            

    def save_model(self, model_name):
        
        return self.model.save(filepath = model_name, format=".h5") 
    
    def load_model(self, model_name):
        return self.model.load(model_name)


#multiclass
"""
ann_object = MyAnn("C:/Users/grace/Code/Python/Courses/ML/csv/iris.csv",
                            target="species",hidden_layer_size=(12,4,-0.25, 3), 
                            activation="relu",loss='categorical_crossentropy', optimizer= 'adam', 
                            batch_size= 32, epochs= 100, monitor= 'accuracy', patience= 5,
                            mode= 'auto', verbose= 1,
                            use_multiprocessing= False)
ann_object.validate_path()
ann_object.valid_df()
ann_object.choose_algarithm()
ann_object.split()
ann_object.scale_data()
ann_object.build_models()
ann_object.predict()
ann_object.evaluate_output()
"""

""" 
# when hidden layer is set to default 
ann_object2 = MyAnn("C:/Users/grace/Code/Python/Courses/ML/csv/iris.csv",
                            target="species",hidden_layer_size=(), 
                            activation="relu",loss='categorical_crossentropy', optimizer= 'adam', 
                            batch_size= 32, epochs= 100, monitor= 'accuracy', patience= 5,
                            mode= 'auto', verbose= 1,
                            use_multiprocessing= False)

                           
ann_object2.validate_path()
ann_object2.valid_df()
ann_object2.choose_algarithm()
ann_object2.preprocess_data()
ann_object2.build_models()
ann_object2.predict()
ann_object2.evaluate_output()
"""
"""
# reg 
ann_object_reg = MyAnn("C:/Users/grace/Code/Python/Courses/ML/csv/Advertising.csv",
                            target="sales",hidden_layer_size=(12, 4, 0, 3), 
                            activation="relu",loss='mse', optimizer= 'adam', 
                            batch_size= 32, epochs= 100, monitor= 'val_loss', patience= 5,
                            mode= 'auto', verbose= 1,
                            use_multiprocessing= False)

ann_object_reg.validate_path()
ann_object_reg.valid_df()
ann_object_reg.choose_algarithm()
ann_object_reg.split()
ann_object_reg.scale_data()
ann_object_reg.build_models()
ann_object_reg.predict()
ann_object_reg.evaluate_output()
# O: General comments
# Over-use of variables in self.
"""
