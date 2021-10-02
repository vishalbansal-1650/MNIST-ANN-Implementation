import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_dataset(data):
    """To prepare and split train-test-validation dataset

    Args:
        data (tf.keras.datasets module): predefined dataset module present in tensorflow.keras

    Returns:
        tuple: returns tuple of x_train,x_valid,x_test,y_train,y_valid,y_test
    """

    logging.info("Preparaing train validation and test dataset")
    (x_train_full,y_train_full), (x_test,y_test) = data.load_data()

    len_train_data = x_train_full.shape[0]     ## total data points in train set
    valid_data_idx = int(len_train_data * 0.1) ## keeping 10% data as validation set

    x_valid, x_train = x_train_full[:valid_data_idx]/255., x_train_full[valid_data_idx:]/255.
    y_valid, y_train = y_train_full[:valid_data_idx], y_train_full[valid_data_idx:]

    return x_train,x_valid,x_test,y_train,y_valid,y_test


def save_model(model,filename):
    """To save the trained model

    Args:
        model (tf.keras model object): trained model
        filename (str): path to save the trained model
    """
    logging.info("Saving the trained model")
    model_dir = 'models'
    os.makedirs(model_dir,exist_ok=True)
    filepath = os.path.join(model_dir,filename)
    model.save(filepath)
    logging.info(f"Trained model saved at path : {filepath}")



def save_plot(model_history,plotname):
    """To generate and save plot

    Args:
        model_history (keras model object): trained model
        plotname (str): path to save the plot
    """

    df = pd.DataFrame(model_history.history)

    df.plot(figsize=(10,8))
    plt.grid(True)

    logging.info("Saving the model performance plot")
    plot_dir = 'plots'
    os.makedirs(plot_dir,exist_ok=True)
    plotpath = os.path.join(plot_dir,plotname)
    plt.savefig(plotpath)
    logging.info(f"Saved plot at path : {plotpath}")


def make_prediction(model,x,y_actual):
    """[To make prediction on dataset]

    Args:
        model (tf.keras model object): trained mode object
        x (numpy array): dataset on which prediction needs to be made
        y_actual (numpy array): actual class labels for x 
    """
    logging.info(f"Making prediction for dataset x: \n{x}")
    y_prob = model.predict(x)
    y_pred = np.argmax(y_prob, axis=-1)
    logging.info(f"Prediction : \n{y_pred}")

    for img_array,pred,actual in zip(x,y_pred,y_actual):
        plt.imshow(X=img_array,cmap='binary')
        plt.axis('off')
        plt.title(f"Predicted : {pred} Actual : {actual}")
        plt.show()
        print("*****"*10)
    
