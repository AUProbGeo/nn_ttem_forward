#Single_forward_comparison.py
#%% [markdown]

#Test and compare GA-AEM vs Neural Network forward modeling speed.
#This script tests one-at-a-time predictions to simulate Metropolis MCMC performance.
#You should be able to run the whole script without problem. 

#%% load modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import numpy as np
import h5py
import datetime
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
Number_of_samples = 100_000  # Number of samples to test
Number_of_samples = 5_000  # Number of samples to test
#parallel = ig.use_parallel(showInfo=1) can be toggled on/off to test if parallelization can be used
#Check if GPU is available for TensorFlow
if tf.config.list_physical_devices('GPU'):
    print("GPU detected. TensorFlow will use the GPU for NN.")
    GPU_available = True
else:
    print("No GPU detected. TensorFlow will use the CPU for NN.")
    GPU_available = False
#%%
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
    if 'D1' in f and 'Nhank' in f['D1'].attrs:
        Nhank = f['D1'].attrs['Nhank']
        Nfreq = f['D1'].attrs['Nfreq']
    else:
        Nhank = 280
        Nfreq = 12
    z = f['M1'].attrs['x'] if 'x' in f['M1'].attrs else f['M1'].attrs['z']
    thickness = np.diff(z)

print("Generating STM files...")
stmfiles, _ = ig.gex_to_stm(file_gex, Nhank=Nhank, Nfreq=Nfreq, showInfo=0)

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
    _ = ig.forward_gaaem(C=1/M_test_comparison[i:i+1, :], thickness=thickness,
                         file_gex=file_gex, stmfiles=stmfiles,
                         Nhank=Nhank, Nfreq=Nfreq, parallel=False, showInfo=-1)

print(f"Starting GA-AEM timing test with {Number_of_samples} samples...")
start_time = datetime.datetime.now()

# Process ONE sample at a time
for i in tqdm(range(Number_of_samples), desc="GA-AEM single forward", dynamic_ncols=True):
    D_single = ig.forward_gaaem(C=1/M_test_comparison[i:i+1, :], thickness=thickness,
                                file_gex=file_gex, stmfiles=stmfiles,
                                Nhank=Nhank, Nfreq=Nfreq, parallel=False, showInfo=-1)
    

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
print(f"Neural Network: {avg_time_nn:.2f} ms per sample. GPU={'Yes' if GPU_available else 'No'}")
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
