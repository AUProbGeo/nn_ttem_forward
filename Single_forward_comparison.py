#Single_forward_comparison.py
#%% [markdown]

#Test and compare GA-AEM vs Neural Network forward modeling speed.
#This script tests one-at-a-time predictions to simulate Metropolis MCMC performance.
#You should be able to run the whole script without problem. 

#%% load modules
import numpy as np
import h5py
import datetime
import os
import tensorflow as tf
from tqdm.auto import tqdm
import integrate as ig

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths

#Set base dir
base_dir = os.path.dirname(os.path.abspath(__file__))


    #File to the models and prior data / forward data 
f_prior_data_general_h5 = os.path.join(base_dir, "daugaard_valley_prior_N2000000.h5")
f_data_h5 = os.path.join(base_dir, "DAUGAARD_AVG.h5")

#model_path to the trained neural network.
model_path = os.path.join(
    base_dir,
    "trained_models",
    "model_big_prior_DG_HL_3_HU_300_CN_0.5_PV_200.h5",
)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

# Test parameters
Number_of_samples = 1000  # Number of samples to test
#parallel = ig.use_parallel(showInfo=1) can be toggled on/off to test if parallelization can be used

#%%
# ============================================================================
# HELPER FUNCTION: GA-AEM SINGLE FORWARD
# ============================================================================

def forward_gaaem_single(M_single, file_gex, f_prior_reference_h5, 
                         Nhank=None, Nfreq=None, showInfo=0):
    """
    Generate GA-AEM forward data for a SINGLE model.
    Matches the behavior of prior_data_gaaem() for single-sample predictions.
    """
    
    # Load Nhank and Nfreq from reference file if not provided
    if Nhank is None or Nfreq is None:
        with h5py.File(f_prior_reference_h5, 'r') as f:
            if 'D1' in f and 'Nhank' in f['D1'].attrs:
                Nhank = f['D1'].attrs['Nhank']
                Nfreq = f['D1'].attrs['Nfreq']    
            else:
                Nhank = 280
                Nfreq = 12
                
    
    # Load depth/thickness information (cached after first call)
    if not hasattr(forward_gaaem_single, 'thickness'):
        with h5py.File(f_prior_reference_h5, 'r') as f:
            # Get depth - EXACTLY like prior_data_gaaem
            if 'x' in f['M1'].attrs:
                z = f['M1'].attrs['x']
            else:
                z = f['M1'].attrs['z']
            
            # Calculate thickness from depth - EXACTLY like prior_data_gaaem
            forward_gaaem_single.thickness = np.diff(z)
    
    thickness = forward_gaaem_single.thickness
    
    # Ensure M_single is 2D array (1, n_layers)
    if M_single.ndim == 1:
        M_single = M_single.reshape(1, -1)
    
    # Convert resistivity to conductivity - EXACTLY like prior_data_gaaem
    C_single = 1 / M_single
    
    # Generate STM files once and cache
    if not hasattr(forward_gaaem_single, 'stmfiles'):
        forward_gaaem_single.stmfiles, _ = ig.gex_to_stm(
            file_gex, Nhank=Nhank, Nfreq=Nfreq, showInfo=showInfo
        )
    
    # Run GA-AEM forward modeling - EXACTLY like prior_data_gaaem
    D_single = ig.forward_gaaem(
        C=C_single, 
        thickness=thickness,
        file_gex=file_gex,
        stmfiles=forward_gaaem_single.stmfiles,
        Nhank=Nhank, 
        Nfreq=Nfreq, 
        parallel=False,  # It seem to be slighly faster to run with False for single forward runs, but you can test with True if you want.
        showInfo=showInfo
    )
    
    # Return as 1D array
    return D_single.flatten()


# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

print("="*70)
print("LOADING DATA AND MODELS")
print("="*70)

# Get GEX file
file_gex = ig.get_gex_file_from_data(f_data_h5)
print(f"Using GEX file: {file_gex}")

# Load test models from prior (in LINEAR space, NOT log10)
print(f"Loading {Number_of_samples} test samples from {f_prior_data_general_h5}...")
with h5py.File(f_prior_data_general_h5, 'r') as f:
    M_test_comparison = f['M1'][0:Number_of_samples, :]  # Linear space
    print(f"M_test shape: {M_test_comparison.shape}")

# Load Neural Network model
print(f"Loading NN model from {model_path}...")
model = tf.keras.models.load_model(model_path)
print("NN model loaded successfully!")

#%%
# ============================================================================
# TEST 1: NEURAL NETWORK ONE-AT-A-TIME
# ============================================================================

print("\n" + "="*70)
print("TEST 1: NEURAL NETWORK ONE-AT-A-TIME")
print("="*70)

#Warm up run, uses GPU
print("Warm-up (10 samples)...")
with tf.device('/GPU:0'):
    for i in tqdm(range(10), desc="NN warm-up", leave=False, dynamic_ncols=True):
        #Take the log10 of the model for the NN input
        M_single_log10 = np.log10(M_test_comparison[i:i+1, :])
        _ = model(M_single_log10, training=False)

print(f"Starting NN timing test with {Number_of_samples} samples...")

start_time = datetime.datetime.now()
# Process one sample at a time
with tf.device('/GPU:0'):
    for i in tqdm(range(Number_of_samples), desc="NN single forward", dynamic_ncols=True):
        M_single_log10 = np.log10(M_test_comparison[i:i+1, :])
        D_pred_single = model(M_single_log10, training=False).numpy()
        D_pred_single = 10**D_pred_single

end_time = datetime.datetime.now()

time_taken_nn = end_time - start_time
avg_time_nn = time_taken_nn.total_seconds() / Number_of_samples * 1000

print(f"\nNEURAL NETWORK RESULTS:")
print(f"  Total time: {time_taken_nn.total_seconds():.2f} seconds")
print(f"  Average per sample: {avg_time_nn:.2f} ms")
print(f"  Estimated for 2M samples: {time_taken_nn.total_seconds() * 2000000 / Number_of_samples / 3600:.2f} hours")

#%%
# ============================================================================
# TEST 2: GA-AEM ONE-AT-A-TIME
# ============================================================================


print("\n" + "="*70)
print("TEST 2: GA-AEM ONE-AT-A-TIME")
print("="*70)

# Warm-up run - 10 samples one-at-a-time
print("Warm-up (10 samples)...")
for i in tqdm(range(10), desc="GA-AEM warm-up", leave=False, dynamic_ncols=True):
    _ = forward_gaaem_single(M_test_comparison[i], file_gex, f_prior_data_general_h5)

print(f"Starting GA-AEM timing test with {Number_of_samples} samples...")
start_time = datetime.datetime.now()

# Process ONE sample at a time
for i in tqdm(range(Number_of_samples), desc="GA-AEM single forward", dynamic_ncols=True):
    D_single = forward_gaaem_single(M_test_comparison[i], file_gex, f_prior_data_general_h5)

end_time = datetime.datetime.now()

time_taken_gaaem = end_time - start_time
avg_time_gaaem = time_taken_gaaem.total_seconds() / Number_of_samples * 1000

print(f"\nGA-AEM RESULTS:")
print(f"  Total time: {time_taken_gaaem.total_seconds():.2f} seconds")
print(f"  Average per sample: {avg_time_gaaem:.2f} ms")
print(f"  Estimated for 2M samples: {time_taken_gaaem.total_seconds() * 2000000 / Number_of_samples / 3600:.2f} hours")

#%%
# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Neural Network: {avg_time_nn:.2f} ms per sample")
print(f"GA-AEM:         {avg_time_gaaem:.2f} ms per sample")
print(f"\nSpeedup: {avg_time_gaaem / avg_time_nn:.1f}x faster with Neural Network")
print(f"\nFor 2 million samples (Metropolis MCMC):")
print(f"  Neural Network: ~{time_taken_nn.total_seconds() * 2000000 / Number_of_samples / 3600:.2f} hours")
print(f"  GA-AEM:         ~{time_taken_gaaem.total_seconds() * 2000000 / Number_of_samples / 3600:.2f} hours")
print(f"  Time saved: ~{(time_taken_gaaem.total_seconds() - time_taken_nn.total_seconds()) * 2000000 / Number_of_samples / 3600:.2f} hours")
print("="*70)

# Clean up
del M_test_comparison
del D_pred_single
del D_single

print("\nTest complete!")
# %%
