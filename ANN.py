"""
author : Vishal Bansal

"""

from utils.all_utils import get_dataset,save_model,save_plot
from utils.model import build_model_architecture, build_model
import logging
import os
import tensorflow as tf

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir='logs'
os.makedirs(logs_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir,"ANN-logs.log"), level=logging.INFO, format=logging_str, filemode='a')

def main(data,LOSS_FUNCTION,OPTIMIZER,METRICS,EPOCHS, BATCH_SIZE, VAL_BATCH_SIZE):
    """main wrapper function to build ANN model and make prediction

    Args:
        data (): [description]
        LOSS_FUNCTION (str): loss function to be used while training
        OPTIMIZER (str): optimizer to be used while training
        METRICS (list): metrics to be used while training
        EPOCHS (number): number of epochs to be used while training
        BATCH_SIZE (number): batch size for train data set
        VAL_BATCH_SIZE (number): validation batch size
    """
    
    x_train,x_valid,x_test,y_train,y_valid,y_test = get_dataset(data)

    n1_input = x_train.shape[1]
    n2_input = x_train.shape[2]
    n_class = len(set(y_train))
    l1_units = 300
    l2_units = 100

    model_clf = build_model_architecture(n1_input, n2_input, n_class, l1_units, l2_units)
    model_history = build_model(model_clf,LOSS_FUNCTION,OPTIMIZER,METRICS,EPOCHS,x_train,y_train,x_valid,y_valid,BATCH_SIZE,VAL_BATCH_SIZE)

    logging.info("Making Prediction on Test Dataset")
    model_clf.evaluate(x=x_test, y=y_test,batch_size=BATCH_SIZE)

    save_model(model_clf,'ANN.h5')
    save_plot(model_history,'ANN-MNIST.png')



if __name__ == "__main__":

    data = tf.keras.datasets.mnist
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]
    EPOCHS = 30
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 32

    try:

        logging.info(">>>>>>>>>>>>>>>>>>>>>>>>> Starting of Script >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        main(data, LOSS_FUNCTION, OPTIMIZER, METRICS, EPOCHS, BATCH_SIZE, VAL_BATCH_SIZE )
        logging.info(" <<<<<<<<<<<<<<<<<<<< Model trained successfully <<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    except Exception as e:
        logging.exception(e)
        raise e 

