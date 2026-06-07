# %% [markdown]
# # Accelerating tTEM forward modelling with a generalizable neural network
# ## Companion script to Mehl, Madsen & Hansen — GEUS Bulletin
#
# This script demonstrates the two-stage workflow described in the manuscript:
#
# - **Stage A** (Sections A–B): Generate a broad, geologically unconstrained general prior,
#   compute tTEM forward responses using the GA-AEM solver, and train the General NN.
#
# - **Stage B** (Sections C–E): Apply the trained General NN as a drop-in forward operator
#   for the geologically informed Daugaard standard and Daugaard valley priors,
#   and run probabilistic inversion using the extended rejection sampler,
#   and visualise the posterior.
#
# The script is designed to run cell-by-cell in VS Code or Spyder.
# Set N_use and N_inv to 2_000_000 for the full-scale results reported in the paper.
# The default values of 50_000 are provided for quick testing.


# %% 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

# %% Imports
import gc
import datetime
import os
import sys
import time
from unittest import skip

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import tensorflow as tf

import integrate as ig
from NN_functions import load_data_from_hdf5, train_model
from error_analysis_functions import analyze_errors
from Generic_prior_functions import (
    load_model_and_predict,
    check_other_prior,
    process_prior_with_nn,
    save_post_results,
)

# Check whether parallel computation is available
parallel = ig.use_parallel(showInfo=1)

hardcopy = True



# %% Add a check to determine if GPU is used with tensorflow
print("TensorFlow version:", tf.__version__)
if tf.config.list_physical_devices("GPU"):
    print("GPU is available. TensorFlow will use the GPU for computations.")
else:
    print("GPU is not available. TensorFlow will use the CPU for computations.")


# %% [markdown]
# ## Configuration
#
# Set the number of prior realizations for each stage of the workflow.
# The values below reproduce the full-scale results reported in the paper.
# To run a quick test, reduce all values to e.g. `50_000`.

# %%
N = 2_000_000  # Number of realizations to use for training and evaluation (B)
N_prior = N  # Reals to generate when building a general prior (A1)
N_use = N  # Reals loaded from the general prior for training (B)
N_inv = N  # Reals loaded from the Informed Daugaard prior (B)
N_reject = N  # Reals used by the extended rejection sampler (D)

# N_prior  = 50_000  # Reals to generatSe when building a general prior (A1)
# N_use    = 50_000  # Reals loaded from the general prior for training (B)
# N_inv    = 50_000  # Reals loaded from the Informed Daugaard prior (B)
# N_reject = 50_000  # Reals used by the extended rejection sampler (D)

use_pretrained_model = False  # Set to True to load a pre-trained General NN and
# skip training (Section C1)
use_precomputed_prior = True  # Set to True to load a pre-computed general prior and
# skip sampling and forward computation (Section A1)

# %% Testing. Optionally select a small number of realizations for a quick test run. Set to False to run the full workflow with the values above.
useTest=False
if useTest:
    N = 2_000_000  # Number of realizations to use for training and evaluation (B)
    N_prior = N
    N_use = N
    N_inv = N
    N_reject = N
    use_pretrained_model = False
    use_precomputed_prior = False
    
# %% [markdown]
# ---
# ## Stage A: Generating the general prior and training the General NN
#
# The General NN is trained on a broad, geologically unconstrained general prior.
# Once trained, it generalises to specific, geologically informed priors without retraining.

# %% [markdown]
# ### A. Select case and locate data files
#
# This section sets up the file paths for the observed tTEM data and the two prior models:
#
# - **General prior** (`f_prior_data_general_h5`): a broad, uninformed prior with 1–9 layers
#   and log-uniform resistivities spanning 1–2500 Ω·m. Used to train the General NN.
#
# - **Informed Daugaard prior** (`f_prior_data_h5`): a geologically informed prior derived
#   from borehole data at the Daugaard site.
#   Two informed Daugaard priors (Daugaard standard and Daugaard valley)
#   are used to evaluate generalisation of the General NN.

# %% Get the data files for the selected case
print("Loading data from Daugaard case...")
showInfo = -1
files = ig.get_case_data(case="DAUGAARD", showInfo=showInfo)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)

# General prior: broad, geologically unconstrained — used for training the General NN
f_prior_data_general_h5 = ig.get_case_data(
    case="DAUGAARD",
    filelist=[
        "PRIOR_UNIFORM_NL_1-9_log-uniform_N2000000_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5"
    ],
    showInfo=showInfo,
)[0]
# Informed Daugaard prior: geologically informed prior — used to evaluate generalisation
f_prior_data_valley_h5 = ig.get_case_data(
    case="DAUGAARD",
    filelist=["daugaard_valley_prior_N2000000.h5"],
    showInfo=showInfo,
)[0]
f_prior_data_standard_h5 = ig.get_case_data(
    case="DAUGAARD",
    filelist=["daugaard_standard_prior_N2000000.h5"],
    showInfo=showInfo,
)[0]

# Select which prior to use for evaluation of the General NN generalisation performance
f_prior_data_h5 = "daugaard_valley_prior_N2000000.h5"
#f_prior_data_h5 = "daugaard_standard_prior_N2000000.h5"

print("Observed data file: %s" % f_data_h5)
print("GEX file: %s" % file_gex)
print("General prior file: %s" % f_prior_data_general_h5)
print("Informed Daugaard prior file: %s" % f_prior_data_h5)

# %% [markdown]
# ### A1. Construct the general prior and compute tTEM forward responses
#
# Prior models **M*** are sampled from the general prior using `ig.prior_model_layered`.
# Each model consists of 90 parameters representing resistivity (Ω·m) at 1 m depth intervals
# from 0 to 90 m. Resistivity values are drawn from a log-uniform distribution spanning
# 1–2500 Ω·m, and the number of layers is drawn uniformly from 1 to 9.
#
# Forward data **D*** are computed using the GA-AEM solver (Brodie, 2020).
# Each forward response comprises 40 dB/dt measurements (V/m²) across the tTEM time gates.
#
# If a pre-computed general prior file already exists, set `f_prior_data_general_h5` above
# and skip this section.

# %%

if len(f_prior_data_general_h5) == 0 or not use_precomputed_prior:

    RHO_min = 1
    RHO_max = 2500
    RHO_dist = "log-uniform"
    NLAY_min = 1
    NLAY_max = 9
    z_max = 89

    # Sample prior models from the general prior
    t0 = time.time()
    f_prior_general_h5 = ig.prior_model_layered(
        N=N_prior,
        lay_dist="uniform",
        z_max=z_max,
        NLAY_min=NLAY_min,
        NLAY_max=NLAY_max,
        HO_dist=RHO_dist,
        RHO_min=RHO_min,
        RHO_max=RHO_max,
        showInfo=1,
    )
    t_prior = time.time()-t0
    print(
        f"Prior sampling: {N_prior} realizations in {t_prior - t0:.2f} s ({N_prior / (t_prior):.0f} realizations/s)"
    )

    # Compute tTEM forward responses using the GA-AEM solver
    t0 = time.time()
    f_prior_data_general_h5 = ig.prior_data_gaaem(
        f_prior_general_h5, file_gex, parallel=parallel, showInfo=0
    )
    t_forward = time.time()-t0
    print(
        f"Time to compute forward responses for the general prior: {t_forward:.2f} s"
    )
    print(f"Forward responses per second: {N_prior / (t_forward):.2f}")

    # ############################
    print("##################### ##################### #####################")
    print("  N_prior = %d" % N_prior)
    print("  prior sampling = %.2f s" % t_prior)
    print("  forward responses = %.2f s" % t_forward)
    print("  Total time = %.2f s" % (t_prior + t_forward))
    print("# # # # #")
    print("  Prior realizations per second: % .2f /s" % (N_prior / (t_prior)))
    print("  Forward responses per second: % .2f /s" % (N_prior / (t_forward)))
    print( "  Total realizations per second: % .2f /s" % (N_prior / (t_prior + t_forward)) )
    print("##################### ##################### #####################")


print("General prior file: %s" % f_prior_data_general_h5)

# %% [markdown]
# ### B. Load data for training and evaluation
#
# Prior models **M*** and forward responses **D*** are loaded from the HDF5 file and
# scaled to base-10 logarithm prior to training. Rows containing negative or NaN values
# are removed. The dataset is split into 80% training, 10% validation, and 10% test sets.
#
# A separate dataset from the Informed Daugaard prior is loaded for out-of-distribution
# evaluation. This prior was not used during training and serves as the key generalisation test.
#
# If a pre-trained General NN is already available, proceed directly to Section C.

# %%
# Load training, validation, and test data from the general prior
M_train, D_train, M_val, D_val, M_test, D_test, N_train, N_val, N_test, Nm, Nd = (
    load_data_from_hdf5(f_prior_data_general_h5, N_use, training=True)
)

# Load data from the Informed Daugaard prior (not used during training)
M_test_detailed, D_test_detailed, Nm_other, Nd_other = load_data_from_hdf5(
    f_prior_data_h5, N_inv, training=False
)

# Names used for labelling saved models and plots
prior_name = "General prior"
other_prior_name = "Informed Daugaard"
prefix = "TEST"
type_model = "HL_3_HU_300_PV_200.h5"

# %% [markdown]
# ### B1. Train the General NN (or load a pre-trained model)
#
# The neural network architecture comprises 90 input nodes (resistivity at 1 m depth intervals),
# three hidden layers of 300 units each with ReLU activation, and 40 output nodes (data parameters).
# Training uses the Adam optimizer with a learning rate of 0.0005 and a batch size of 4 096.
# Early stopping is applied with a patience of 200 epochs to prevent overfitting.
# Gradient clipping is used to stabilise training.


#
# A pre-trained General NN can be loaded directly to skip the training step above.
# The model path should point to the saved `.h5` file.

# %%
# Load the trained General NN and evaluate on the general prior test set
if use_pretrained_model:
    # Set the path to the pre-trained model .h5 file
    model_h5 = os.path.join(
        "trained_models", "model_big_prior_DG_HL_3_HU_300_CN_0.5_PV_200.h5"
    )

    model, D_pred, best_val_loss, plots_dir = load_model_and_predict(
        model_h5,
        M_test,
        D_test,
        make_dir=True,
        prefix=prefix,
        prior_name=prior_name,
        other_prior_name=other_prior_name,
        make_plots=False,
    )

else:

    learning_rate = 0.0005  # Adam optimizer learning rate
    batch_size = 4096  # Mini-batch size
    nunits = 300  # Units per hidden layer
    nhidden = 3  # Number of hidden layers
    activation_function = "relu"  # Hidden layer activation function
    epochs = 2000  # Maximum number of training epochs (set to 2500 for full training)
    clipnorm_value = 0.5  # Gradient clipping norm

    use_early_stopping = True  # Stop training when validation loss stops improving
    patience_value = 200  # Number of epochs without improvement before stopping

    # Train the General NN and evaluate on the held-out test set from the general prior.
    t0 = time.time()
    model, results, D_pred, plots_dir, log_dir, model_h5 = train_model(
        prior_name,
        prefix,
        learning_rate,
        batch_size,
        nunits,
        nhidden,
        activation_function,
        epochs,
        clipnorm_value,
        Nm,
        Nd,
        M_train,
        D_train,
        M_val,
        D_val,
        M_test,
        D_test,
        use_early_stopping,
        patience_value,
    )
    t_train = time.time() - t0
    print("Training: % .2f s" % t_train)

    # Retrieve the best validation loss achieved during training
    best_val_loss = results["val_loss"]

# %% [markdown]
# ### B2. Evaluate accuracy of the General NN
#
# Model accuracy is evaluated by computing the relative error between NN predictions and the
# corresponding physics-based GA-AEM forward responses, following the approach in the paper.
#
# The evaluation is performed on two datasets:
# 1. The held-out test set from the general prior (in-distribution)
# 2. The Informed Daugaard prior (out-of-distribution — the key generalisation test)

# %%
# Accuracy on the uninformed test set (in-distribution)
metrics_from_model = analyze_errors(
    D_test,
    D_pred,
    save_dir=False,
    best_val_loss=None,
    error_threshold=0.05,
    name_of_test_prior="Uninformed test set",
    title=None,
    make_pdf=False,
)

# %%
# Accuracy on the Informed Daugaard prior (out-of-distribution)
# This prior was not used during training and serves as the key generalisation test.
t0 = time.time()
D_pred_detailed = model.predict(M_test_detailed, batch_size=100000, verbose=1)
t_pred = time.time()-t0
print(
    f"NN prediction: {M_test_detailed.shape[0]} forward evals in {t_pred:.2f} s "
    f"({M_test_detailed.shape[0] / (t_pred):.0f} evals/s)"
)
metrics_on_informed_prior = analyze_errors(
    D_test_detailed,
    D_pred_detailed,
    save_dir=False,
    best_val_loss=None,
    error_threshold=0.05,
    name_of_test_prior="Informed Daugaard prior",
    title=None,
    make_pdf=False,
)

# %% Print some timing results for reference
import os
import platform
import psutil

# Print to screen som information and the system, CPU, memory, and GPU info if available
print("##################### ##################### #####################")
print("##################### SYSTEM INFO          #####################")
print("  OS       : %s %s" % (platform.system(), platform.release()))
print("  Python   : %s" % platform.python_version())
print("  CPU      : %s" % platform.processor())
print("  CPU cores: %d physical / %d logical" % (psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)))
mem = psutil.virtual_memory()
print("  RAM      : %.1f GB total / %.1f GB available" % (mem.total / 1e9, mem.available / 1e9))
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get('device_name', gpu.name)
            print("  GPU[%d]   : %s" % (i, name))
    else:
        print("  GPU      : none (no CUDA GPU detected by TensorFlow)")
except ImportError:
    print("  GPU      : tensorflow not available")
print("##################### ##################### #####################")


try:
    # ############################
    print("##################### ##################### #####################")
    print("##################### SETUP                #####################")
    print("  N_prior = %d" % N_prior)
    print("  prior sampling = %.2f s" % t_prior)
    print("  forward responses = %.2f s" % t_forward)
    print("  Total time = %.2f s" % (t_prior + t_forward))
    print("# # # # #")
    print("  Prior realizations per second: % .2f /s" % (N_prior / (t_prior)))
    print("  Forward responses per second: % .2f /s" % (N_prior / (t_forward)))
    print( "  Total realizations per second: % .2f /s" % (N_prior / (t_prior + t_forward)) )
    print("##################### ##################### #####################")
except:
    pass

try:
    # ############################
    print("##################### ##################### #####################")
    print("##################### NN train/predict      #####################")
    print("  N_use = %d" % N_use)
    print("  train time = %.2f s" % t_train)
    print("  N_test = %d" % M_test_detailed.shape[0])
    print("  prediction time = %.2f s" % t_pred)
    print("  Total time = %.2f s" % (t_train + t_pred))
    print("# # # # #")
    print("  Training realizations per second: % .2f /s" % (N_use / (t_train)))
    print("  Prediction evals per second: % .2f /s" % (M_test_detailed.shape[0] / (t_pred)))
    print( "  Total realizations per second: % .2f /s" % (N_use / (t_train + t_pred)) )
    print("##################### ##################### #####################")
except:
    pass

try:
    # ############################
    print("##################### ##################### #####################")
    print("##################### Speedup               #####################")
    print("  GA-AEM forward responses for %d realizations: %.2f s" % (N_use, t_forward))
    print("  General NN predictions for %d realizations: %.2f s" % (M_test_detailed.shape[0], t_pred))
    speedup = t_forward / t_pred
    print("  Speedup: %.0f times faster than GA-AEM" % speedup)
    print("##################### ##################### #####################")
except:
    pass


# %% 
import sys
sys.exit(0)

# %% [markdown]
# ---
# ## Stage B: Generalisation — applying the General NN to an informed prior
#
# The trained General NN is used as a drop-in replacement for the GA-AEM solver within the
# INTEGRATE framework. No retraining is required when applying the General NN to a new,
# geologically informed prior built on the same tTEM system geometry.

# %% [markdown]
# ### C. Visualise generalisation of the General NN to the Informed Daugaard prior
# %%
# Evaluate generalisation to the Informed Daugaard prior
check_other_prior(
    prior_name,
    other_prior_name,
    model,
    M_test_detailed,
    D_test_detailed,
    plots_dir,
    best_val_loss,
    make_plots=True,
)


# %% [markdown]
# ### C1. Compare performance across data splits
#
# The trained NN is evaluated on all data splits (training, validation, test, and informed prior)
# by comparing NN predictions to reference GA-AEM forward responses.
# Loss is reported as MSE in log₁₀ space, consistent with the training objective.

# %%
# Evaluate NN on all data splits and compare to reference forward data
splits = [
    ("TRAINING", M_train, D_train),
    ("BLIND (val)", M_val, D_val),
    ("TEST (general prior)", M_test, D_test),
    ("INFORMED (Daugaard)", M_test_detailed, D_test_detailed),
]

rows = []
for name, M, D in splits:
    loss = model.evaluate(M, D, batch_size=100000, verbose=0)
    rows.append({"Split": name, "MSE (log-space)": loss})

table = pd.DataFrame(rows)
print(table)

# %% [markdown]
# ### C2. Benchmark forward computation speed
#
# The General NN generates tTEM forward responses for a large number of prior realizations
# and records the computation time. The benchmark is repeated multiple times to assess
# stability, with an optional warm-up pass to initialise the GPU.
#
# For reference, computing forward responses for 2 million realizations with the GA-AEM
# solver took 11.5 hours on the same laptop. The General NN reduces this to 1–3 seconds,
# corresponding to a speedup of approximately 710–1 890 times.

# %%
use_warmup = True  # Perform a warm-up pass to initialise the GPU before timing
number_of_times_to_predict = 10  # Number of timed repetitions

forward_data_time_results = []

if use_warmup:
    print("Warming up...")
    _ = model.predict(M_test_detailed[:1000], batch_size=100000, verbose=0)
    del _
    print("Warm-up complete. Starting timed predictions.")

for i in range(number_of_times_to_predict):
    if i > 0:
        tf.keras.backend.clear_session()
        gc.collect()

    start_time = datetime.datetime.now()
    D_pred_timed = model.predict(M_test_detailed, batch_size=100000, verbose=1)
    intermediate_time = datetime.datetime.now()

    # Transform predictions from log₁₀ space back to linear space
    D_pred_timed = 10**D_pred_timed

    end_time = datetime.datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    time_unscaling = (end_time - intermediate_time).total_seconds()

    print(
        f"Repetition {i+1}/{number_of_times_to_predict}: "
        f"{M_test_detailed.shape[0]} samples in {time_taken:.2f} s "
        f"(log-to-linear transform: {time_unscaling:.4f} s)"
    )
    forward_data_time_results.append(time_taken)
    del D_pred_timed

print(f"\nAverage time: {np.mean(forward_data_time_results):.2f} s")
print(f"Fastest:      {np.min(forward_data_time_results):.2f} s")
print(f"Slowest:      {np.max(forward_data_time_results):.2f} s")

# %% [markdown]
# ### D. Probabilistic inversion using the extended rejection sampler
#
# The General NN is used to replace GA-AEM forward responses in the Informed Daugaard prior.
# The function `process_prior_with_nn` creates a new HDF5 file containing the same prior
# models **M'*** but with forward responses **D'*** computed by the General NN rather than
# GA-AEM. Inversion is then performed using the extended rejection sampler as implemented
# in the INTEGRATE framework (Hansen, 2020).
#
# Two inversions are compared:
# - **Reference inversion**: uses the Informed Daugaard prior with GA-AEM forward responses
# - **General NN inversion**: uses the same prior models with NN-predicted forward responses

# %%
# Replace GA-AEM forward responses in the Informed Daugaard prior with General NN predictions
f_prior_data_h5_org, f_prior_data_h5_nn, D_new, D_org = process_prior_with_nn(
    f_prior_data_h5, model, f_data_h5, batch_size=100000, verbose=1, plot_results=True
)


# %%
# Reference inversion: Informed Daugaard prior with GA-AEM forward responses
f_post_h5_org = ig.integrate_rejection(
    f_prior_data_h5_org, f_data_h5, N_use=N_reject, parallel=parallel, showInfo=1
)

# %%
# General NN inversion: Informed Daugaard prior with NN-predicted forward responses
f_post_h5_nn = ig.integrate_rejection(
    f_prior_data_h5_nn, f_data_h5, N_use=N_reject, parallel=parallel, showInfo=1
)

# %%
# Save posteriors to the inversion_data directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_post_results(f_post_h5_nn, f"Informed_Daugaard_NN_{timestamp}.h5")

# %%
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_post_results(f_post_h5_org, f"Informed_Daugaard_reference_{timestamp}.h5")

# %% [markdown]
# ### E. Plot the posterior along a profile with survey geometry
#
# The posterior is plotted along a profile defined by two endpoints in UTM coordinates.
# Sounding locations within a given buffer distance of the profile are selected and
# displayed in the geometry of the Daugaard tTEM survey.

# %%

f_data_h5 = "DAUGAARD_AVG.h5"
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# Define profile endpoints in UTM coordinates (Easting, Northing)
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
buffer = 10.0

# Find sounding locations within the buffer distance of the profile
id_line, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)

# Plot the survey area and highlight the selected profile
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, c=ELEVATION, s=10, cmap="viridis")
plt.colorbar(label="Elevation (m)")
plt.plot(X[id_line], Y[id_line], "r.", markersize=8, label="Profile", zorder=2)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the posterior along the selected profile
# print('Plotting posterior along profile')
# ig.plot_profile(f_post_h5_loaded, ii=id_line, gap_threshold=50, xaxis='y')

# %%
# Plot the posterior median resistivity along the selected profile

print("Plotting org posterior along profile")
ig.plot_profile_continuous(
    f_post_h5_org,
    ii=id_line,
    gap_threshold=50,
    xaxis="y",
    panels=["median"],
    show_data=True,
)

print("Plotting nn posterior along profile")
ig.plot_profile_continuous(
    f_post_h5_nn,
    ii=id_line,
    gap_threshold=50,
    xaxis="y",
    panels=["median"],
    show_data=True,
)


# %% Plot full profiles
plot_full_profiles = False  # Set to True to plot the full posterior along the profile, including uncertainty
if plot_full_profiles:
    print("Plotting full org posterior along profile")
    ig.plot_profile(
        f_post_h5_org, ii=id_line, gap_threshold=50, xaxis="y", show_data=True
    )

    print("Plotting full nn posterior along profile")
    ig.plot_profile(
        f_post_h5_nn, ii=id_line, gap_threshold=50, xaxis="y", show_data=True
    )


# %%
