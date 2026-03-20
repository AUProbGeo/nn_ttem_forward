#NN functions
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import datetime
import os
import pandas as pd
import time
import gc



def load_trained_model(model_path):
    """
    Load a previously trained neural network model from an H5 file.
    
    Parameters:
    -----------
    model_path : str
        Path to the H5 file containing the saved model.
        
    Returns:
    --------
    model : tf.keras.models.Sequential
        The loaded model.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        # Print model summary
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def batch_process_model(model, M_test, batch_size=400000, verbose=1):
    """
    Process data in batches using model().numpy() and optionally return the complete output.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The TensorFlow model to use for predictions
    M_test : numpy.ndarray
        The input data to process (samples × features)
    batch_size : int
        Size of each batch for processing
    verbose : int
        0 = silent, 1 = progress info
        
    Returns:
    --------
    final_output : numpy.ndarray
        The complete output after processing all batches
    """
    import time
    import numpy as np
    import gc
    start_time = time.time()
    # Get total number of samples from the input data
    N_total = len(M_test)
    
    # Print configuration information if verbose mode is enabled
    if verbose:
        print(f"Processing {N_total:,} samples with batch_size={batch_size:,}")
        
    
    # Calculate number of batches for reporting
    num_batches = N_total // batch_size + (1 if N_total % batch_size > 0 else 0)
    
    # Start timing the operation
    # No warm-up is performed per request - this measures "cold start" performance
    
    
    # Use a list to collect batch outputs - more memory efficient than pre-allocation
    # Only collect outputs if we need to return them
    batch_outputs = [] 
    
    # Process data in batches to handle large datasets
    for i in range(0, N_total, batch_size):
        # Calculate end index for current batch (handles final partial batch)
        end_idx = min(i + batch_size, N_total)
        
        # Report progress for current batch
        if verbose:
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num}/{num_batches}: samples {i:,} to {end_idx:,}")
        
        # Get the current batch of data
        M_batch = M_test[i:end_idx]
        
        # Generate predictions for this batch
        # model(input) calls the model directly, .numpy() converts TensorFlow tensor to NumPy array
        D_batch = model(M_batch).numpy()
        
        #Append the batch output to the list
        batch_outputs.append(D_batch)
        del D_batch
        
    # np.concatenate joins all batch outputs along the first dimension (samples)
    final_output = np.concatenate(batch_outputs, axis=0)
    # Clean up the list of individual batches to save memory
    del batch_outputs
    
    # Calculate total elapsed time and throughput metrics
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = N_total / elapsed_time
    
    # Report performance metrics if in verbose mode

    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} samples/second")
    
    # Force garbage collection to clean up memory
    gc.collect()
    
    return final_output
    



def load_data_from_hdf5(f_data_name,N_samples,training=True):
    """
    Load data from an HDF5 file.
    
    Parameters:
    -----------
    f_data_name : str
        Path to the HDF5 file.
    N_samples : int
        Number of samples to load.

    training : bool
        If True, split the data into training, validation, and test sets.
        if False, load the data for testing only.
    
    """
    import os
    import h5py
    import numpy as np

    with h5py.File(f_data_name, 'r') as f:
        M = f['M1'][:, ::1]
        D = f['D1'][:, ::1]

    # log transform the model parameters
    M = np.log10(M)
    # log transform the data, and take the real part to avoid complex numbers
    D = np.real(np.log10(D))
    # the index of all data in D that are not NaN
    idx = np.where(~np.isnan(D).any(axis=1))[0]
    
    M = M[idx]
    D = D[idx]

    # Ensure we don't exceed the requested number of samples
    N = min(len(M), N_samples)
    M = M[:N]
    D = D[:N]

    # %%
    #Find the shape of the data
    N, Nm = M.shape #Here N is the number of samples, Nm is the number of model parameters
    N, Nd = D.shape #Here N is the number of samples, Nd is the number of data points


    if training==True:
        
        # split data into training, validation, and test data
        N_train = int(N * 0.8)
        N_val = int(N * 0.1)
        N_test = N - N_train - N_val

        M_train = M[:N_train]
        D_train = D[:N_train]
                  
        M_val = M[N_train:N_train + N_val]
        D_val = D[N_train:N_train + N_val]

        M_test = M[N_train + N_val:]
        D_test = D[N_train + N_val:]

        return M_train, D_train, M_val, D_val, M_test, D_test, N_train, N_val, N_test, Nm, Nd
    else:
        M_test_other = M[:N_samples]
        D_test_other = D[:N_samples]
        return M_test_other, D_test_other, Nm, Nd
    
   









#####################################################################################


def train_model(prior_name, prefix,learning_rate = 0.003, batch_size = 500,  nunits = 300,  nhidden = 3,  activation_function = 'relu',   
                epochs = 2000, clipnorm_value = 0.5,Nm=None,Nd=None, M_train=None, D_train=None,
                M_val=None, D_val=None,M_test=None, D_test=None,
                use_early_stopping = True,  # Set to True to use early stopping
                patience_value = 200):  # Patience value for early stopping
    """
    Train a neural network model using TensorFlow.
    Saves the best modelweights during training and evaluates the model on a test set.
    also saves the results and plots to specified directories.
    Parameters
    ----------
    prior_name : str
        A name for the prior used, included in saved file names.
    prefix : str
        A prefix for directory names where logs and plots will be saved.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training.
    nunits : int
        Number of units in each hidden layer.
    nhidden : int
        Number of hidden layers.
    activation_function : str
        Activation function to use in hidden layers (e.g., 'relu', 'tanh').
    epochs : int
        Maximum number of training epochs.                    
    clipnorm_value : float or False
        If a float, applies gradient clipping with the specified norm value. If False, no clipping is applied.
    Nm : int
        Number of model parameters (input features).            
    Nd : int
        Number of data points (output features). 
    M_train : numpy.ndarray
        Training input data (samples × features).
    D_train : numpy.ndarray
        Training output data (samples × features).
    M_val : numpy.ndarray
        Validation input data (samples × features). 
    D_val : numpy.ndarray
        Validation output data (samples × features).              
    M_test : numpy.ndarray
        Test input data (samples × features).
    D_test : numpy.ndarray
        Test output data (samples × features).

    use_early_stopping : bool
        If True, uses early stopping based on validation loss.
    patience_value : int
        Number of epochs with no improvement after which training will be stopped if early stopping is enabled. 
    Returns
    ------- 
    model : tf.keras.Model
        The trained model with the best weights loaded.                         
    results : dict
        A dictionary containing training results and metrics.
    D_pred : numpy.ndarray
        The model's predictions on the test set.
    plots_dir : str
        Directory where plots were saved.
    log_dir : str
        Directory where logs were saved.


    """
    # Create a base name for all files
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = (f"prior_{prior_name}_"
                                    f"CN_{clipnorm_value}_"
                                   f"pv_{patience_value}_"
                                   f"lr_{learning_rate}_bs_{batch_size}_"
                                   f"units_{nunits}_hidden_{nhidden}_"
                                   f"epochs_{epochs}_"
                                   f"{timestamp}")
    
    # Create directories
    log_dir = os.path.join(prefix, base_name)
    plots_dir = os.path.join('plots',prefix, base_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Define the model
    model = tf.keras.Sequential()
    #Add input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(Nm,)))
    #Add hidden layers
    for i in range(nhidden):
        model.add(tf.keras.layers.Dense(nunits, activation=activation_function))
    #add output layer
    model.add(tf.keras.layers.Dense(Nd))

    

    model.summary()
    print(f'the number of hi')

    #Set the optimeizer
    if clipnorm_value !=False:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm_value)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Define the list of callbacks
    callbacks = [tensorboard_callback]  # Makes a list with callback and puts the tensorboard callback in it
    if use_early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_value, restore_best_weights=True)
        callbacks.append(early_stopping_callback)  # Puts the callback from early stopping in the list

    #saves the best model during training:
    # Save the best model weights during training
    checkpoint_filepath = os.path.join(log_dir, 'best_model.weights.h5') #The name of the model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback) #adds to callback list

    #Train the model
    history = model.fit(M_train, D_train, epochs=epochs, batch_size=batch_size,
                         validation_data=(M_val, D_val), callbacks=callbacks,verbose=1)
    
    # Extract the minimum validation loss
    val_loss = min(history.history['val_loss'])

    #Load the the best model after training
    model.load_weights(checkpoint_filepath)


    # Evaluate the model on the test set
    # Ensure M_test is a TensorFlow tensor
    
    test_loss = model.evaluate(M_test, D_test, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    D_pred = model.predict(M_test,batch_size=100000,verbose=1)
    print(f'D_pred shape: {D_pred.shape}')

    mse = np.mean((D_test - D_pred) ** 2)
    rel_err_linear= np.mean(np.abs(10**D_test - 10**D_pred) / np.abs(10**D_test))

    results={'learning_rate': learning_rate,'batch_size': batch_size,'nunits': nunits,'nhidden': nhidden,
                            'activation': activation_function, 'epochs': epochs,'val_loss': val_loss,
                            'test_loss': test_loss,'mse': mse,'rel_err_linear': rel_err_linear
                        }
    
    

    # Save the results to a CSV file
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(plots_dir, f'results_from_the_trained_model {timestamp}.csv'), index=False)

    # Save plots 
    #Plot the predicted vs true values
    plt.figure()
    plt.plot(D_test, D_pred, 'k.', markersize=0.4)
    plt.xlabel('True D')
    plt.ylabel('Predicted D')
    plt.axis('equal')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, 'predicted_vs_true.png'))
    plt.show()
    plt.close()

    #Plot the D_pred vs D_test for the first 30 samples
    plt.figure()
    plt.plot(D_test[0:30], 'k-')
    plt.plot(D_pred[0:30], 'r:')
    plt.xlabel('Sample #')
    plt.ylabel('D')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, f'sample_comparison_0_30_.png'))
    plt.show()

# %%
    ##Plot the D_pred vs D_test for the first sample
    plt.plot(D_test[0], 'k-')
    plt.plot(D_pred[0], 'r:')
    plt.xlabel('Sample #')
    plt.ylabel('D')
    plt.grid()
    plt.savefig(os.path.join(plots_dir, f'sample_comparison_0.png'))
    plt.show()

    model_h5 = os.path.join(log_dir, f'model_{prior_name}_{timestamp}.h5')
    model.save(model_h5)

    return model, results, D_pred, plots_dir, log_dir, model_h5

