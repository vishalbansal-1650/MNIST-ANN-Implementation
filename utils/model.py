"""
author : Vishal Bansal

"""


import logging
import tensorflow as tf
import pandas as pd
import io

def build_model_architecture(n1_input,n2_input,n_class,l1_units,l2_units):
    """ To define model architecture for ANN

    Args:
        n1_input (number): input array.shape[1]
        n2_input (number): input array.shape[2]
        n_class (number): number of classes
        l1_units (number): neuron in layer 1
        l2_units (number): neuron in layer 2

    Returns:
        [tf.keras model object]: Sequential Model architecture object for ANN 
    """

    logging.info("Defining the model architecture")
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[n1_input,n2_input],name='inputlayer1'),
        tf.keras.layers.Dense(units=l1_units, activation='relu', name='hiddenlayer1'),
        tf.keras.layers.Dense(units=l2_units, activation='relu', name='hiddenlayer2'),
        tf.keras.layers.Dense(units=n_class, activation='softmax', name='outputlayer')
    ]
    
    def _get_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str
    model_clf = tf.keras.models.Sequential(layers=LAYERS)
    logging.info(f"Model architecture: \n{_get_model_summary(model_clf)}")
    

    return model_clf


def train_model(model_clf,LOSS_FUNCTION,OPTIMIZER,METRICS,EPOCHS,x_train,y_train,x_valid,y_valid,BATCH_SIZE,VAL_BATCH_SIZE):
    """to fit model on train datset 

    Args:
        model_clf (tf.keras model object): model object
        LOSS_FUNCTION (str): loss function to be used while training
        OPTIMIZER (str): optimizer to be used while training
        METRICS (list): metrics to be used while training
        EPOCHS (number): number of epochs to be used while training
        x_train (numpy array): x_train
        y_train (numpy array): y_train
        x_valid (numpy array): x_valid
        y_valid (numpy array): y_valid
        BATCH_SIZE (number): batch size for train data set
        VAL_BATCH_SIZE (number): validation batch size

    Returns:
        [tf.keras model object]: model.history object 
    """
    logging.info("Compiling the model") 
    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    logging.info("Model Compiled successfully") 

    VALIDATION = (x_valid, y_valid)

    logging.info("Training the model on train data set") 
    model_history = model_clf.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_data=VALIDATION,batch_size=BATCH_SIZE, validation_batch_size=VAL_BATCH_SIZE)

    logging.info(f"Model training details: \n{pd.DataFrame(model_history.history)}")
    logging.info("Model Trained successfully")

    return model_history



