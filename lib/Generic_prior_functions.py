#Generic_prior_functions

#This function is used to save inversion data after it as been passed into integrate_rejection
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import os
import sys
import shutil
import tensorflow as tf
import pandas as pd

from error_analysis_functions import (analyze_errors, plot_percentage_below_threshold,
                                      analyze_standard_deviation)
from Subplot_function import create_subplots

def save_post_results(f_post_h5, file_name):
    """
    Save or move the inversion results file to the 'inversion_data' directory.

    Parameters:
    - f_post_h5: The file path of the inversion results.
    - file_name: The new name for the output file (without directory path).
    """
    # Ensure the 'inversion_data' directory exists
    save_dir = 'inversion_data'
    os.makedirs(save_dir, exist_ok=True)

    # Full path to the output file
    output_file = os.path.join(save_dir, file_name)

    # Copy or rename the file
    if os.path.isfile(f_post_h5):
        shutil.copy(f_post_h5, output_file)
        print(f"Inversion results saved to {output_file}")
    else:
        raise FileNotFoundError(f"The file {f_post_h5} does not exist.")


#
#This functiosn loads a NN model and makes D_pred while also creating a plot_dir
def load_model_and_predict(model_path, M_test,D_test, make_dir=True,
                            prefix='test', prior_name='prior', other_prior_name='other_prior', make_plots=True):
    '''
    Load a pre-trained model and make predictions on the test data.
    Parameters:
    - model_path: Path to the pre-trained model.
    - M_test: Test data for making predictions.
    - D_test: Test data for error analysis.
    - make_dir: Boolean flag to create a directory for plots.
    - prefix: Prefix for the directory name.
    - prior_name: Name of the prior for labeling.
    - other_prior_name: Name of the other prior for labeling.
    - make_plots: Boolean flag to create plots.


    '''
    # try to time different parts of the code
    # Load the model from the specified path
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return None
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        # Print model summary
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")

    #Sets best validation loss to None
    best_val_loss = None

    #This model is used to make predictions on the data
    # Make predictions using the loaded model

    D_pred = model.predict(M_test,100000,verbose=1)  # Predict using the loaded model

    print("Predictions made successfully!")
    if make_dir == True: 
        #Make a plots dir
        plots_dir = os.path.join('plots',prefix,f'{prior_name} on {other_prior_name}')
        os.makedirs(plots_dir, exist_ok=True)
    
    if make_plots == True:
        #A way to find estimate error in the model
        metrics = analyze_errors(
        D_test,  # Your test data in log10 space
        D_pred,  # Your predictions in log10 space
        plots_dir,  # Directory for saving plots
        best_val_loss, # Best validation loss
        0.05,  # Threshold for relative error analysis
        prior_name
        )
    
        
        plot_percentage_below_threshold(D_test, D_pred, error_threshold=0.05, save_dir=plots_dir, 
                                     show_plot=True, fig_size=(12, 6), title=f'Percentage of Predictions Below Threshold for {prior_name}', 
                                     filename=f'percentage_below_threshold_per_gate {prior_name}.png'
    )

        #Now the functions that makes the subplots:
        #The functions makes subplots for the relative error distributiosn for each gate and saves them in the plots_dir.

        noise_factor=0.05
        # After your model predictions are ready
        title = f'Relative Error Distribution of Neural Network Predictions \nfor {prior_name}'
        savename=f'{prior_name}'
        create_subplots(D_test, D_pred, noise_factor, save_dir=plots_dir,title=title,savename=savename)
    return model, D_pred, best_val_loss, plots_dir

def check_other_prior(prior_name, other_prior_name, model, M_test_other,
                       D_test_other, plots_dir, best_val_loss, make_plots=True, ):
    """
    Check the performance of the model on another prior and create plots for error analysis

    Makes a bunch of plots!


    Parameters:
    - prior_name: Name of the first prior.
    - other_prior_name: Name of the second prior to compare with the first.
    - model: The neural network model to use for predictions.
    - M_test_other: Test data for the other prior model.
    - D_test_other: True values for the other prior model.
    - plots_dir: Directory to save the plots.
    - best_val_loss: Best validation loss from the model training.
    - make_plots: Boolean flag to indicate whether to create plots.


   
    """
    #Make predictions using the loaded model
    D_pred_other = model.predict(M_test_other,100000,verbose=1)  # Predict using the loaded model
    print(D_pred_other.shape)

    
    if make_plots == True:
            #A way to find estimate error in the model on the other prior
        #The standard deviation of the predicted values for the other model
        standard_deviations = analyze_standard_deviation(
            D_test_other,         # True values in from the other prior model
            D_pred_other,         # Predicted values in log space
            save_dir=plots_dir,
            show_plot=True,
            print_errors=True,
            title=f"Standard deviation for {prior_name} on {other_prior_name} Model",
            filename=f"std_for_rel_errors_on {other_prior_name} prior model"

        )
    # After your model predictions are ready
        test_name = f'{other_prior_name}'  # Name of the other prior model
        metrics = analyze_errors(
            D_test_other,  # Your test data in log10 space
            D_pred_other,  # Your predictions in log10 space
            plots_dir,  # Directory for saving plots
            best_val_loss,  # Best validation loss
            0.05,  # Threshold for relative error analysis
            test_name #Name of the other prior model
        )

        #And lastly the subplots for the other model
        #This function makes the 40 subplots
        noise_factor=0.05
        # After your model predictions are ready
        title = f'Relative Error Distribution of Neural Network Predictions \nfor {prior_name} on {other_prior_name}'
        savename=f'{prior_name}_on_{other_prior_name}'
        create_subplots(D_test_other, D_pred_other, noise_factor,
                    plots_dir,title,savename)



        #And lastly the percentage of predictions below threshold for the other model, virkker ikke lige nu
        #import importlib
        #import error_analysis_functions 
        #importlib.reload(error_analysis_functions)
        #from error_analysis_functions import plot_percentage_below_threshold

    # plot_percentage_below_threshold(D_test_other, D_pred_other, error_threshold=0.05, save_dir=plots_dir, 
        #                                 show_plot=True, fig_size=(12, 6), title=f'Percentage of Predictions Below Threshold for {prior_name} on {other_prior_name}', 
        #                                 filename=f'percentage_below_threshold_per_gate {prior_name} on {other_prior_name}.png'
        #)
    return D_pred_other



import integrate as ig
def process_prior_with_nn(f_prior_data_h5, model, f_data_h5, 
                          org_suffix='_org', nn_suffix='_nn', 
                          batch_size=100000, verbose=1, plot_results=True):
    """
    Process a prior dataset with a neural network model.
    
    Parameters:
    -----------
    f_prior_data_h5 : str
        Path to the original prior data H5 file
    model : tensorflow.keras.Model
        Trained neural network model for prediction
    f_data_h5 : str
        Path to the data file needed for plotting
    org_suffix : str, optional
        Suffix for original data file copy
    nn_suffix : str, optional
        Suffix for neural network predicted data file
    batch_size : int, optional
        Batch size for model prediction
    verbose : int, optional
        Verbosity level for model.predict
    plot_results : bool, optional
        Whether to plot the results
        
    Returns:
    --------
    tuple
        (f_prior_data_h5_org, f_prior_data_h5_nn, D_new)
        Paths to original and NN files, plus the new predictions
    """

    #Try to time different parts of the code
    import time
    time_start = time.time()
    f_prior_data_h5_org = 'f_prior_data_org.h5'
    ig.copy_hdf5_file(f_prior_data_h5,f_prior_data_h5_org)
    f_prior_data_h5_nn = 'f_prior_data_nn.h5'
    ig.copy_hdf5_file(f_prior_data_h5,f_prior_data_h5_nn)
    time_now = time.time()
    print(f"Time taken to copy files: {time_now - time_start:.2f} seconds")

    time_start = time.time() 
    # Load M data from the h5 file
    with h5py.File(f_prior_data_h5_org, 'r') as f:
        M_prior = f['M1'][:] 
        D_org   = f['D1'][:]
    
    # Normalize M by taking log10
    M_prior = np.log10(M_prior)
    
    # Pass data through the model to get predictions
    D_new = model.predict(M_prior, batch_size, verbose=verbose)
    
    # Denormalize the predicted data
    D_new = 10**D_new

    time_now = time.time()
    print(f"Time taken for loading and model prediction: {time_now - time_start:.2f} seconds")


    # Save the predicted data to the new file
    ig.save_prior_data(f_prior_data_h5_nn, D_new, id=1, force_delete=True)
    time_now2 = time.time()
    print(f"Time taken to save data: {time_now2 - time_now:.2f} seconds")
    # Plot the results if requested
    if plot_results:
        print("Plotting for f_prior_data_h5_org")
        ig.plot_data_prior(f_prior_data_h5_org, f_data_h5, id=1)
        print("Plotting for f_prior_data_h5_nn")
        ig.plot_data_prior(f_prior_data_h5_nn, f_data_h5, id=1)
        plt.show()
    
    return f_prior_data_h5_org, f_prior_data_h5_nn, D_new, D_org
# %%
