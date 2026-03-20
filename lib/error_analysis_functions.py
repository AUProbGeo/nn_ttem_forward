import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def analyze_errors(D_test, D_pred, save_dir=False,
                    best_val_loss=None, error_threshold=0.05, name_of_test_prior='Test',title=None, make_pdf=False):
    """
    Calculate and visualize error metrics in both linear and log space.
    
    Args:
        D_test: True values in log10 space
        D_pred: Predicted values in log10 space
        save_dir: Directory to save plots and results
        best_val_loss: Best validation loss from training (optional)
        error_threshold: Threshold for relative error analysis (default: 0.05 for 5%)
        name_of_test_prior: Name of the test prior (default: 'Test')
        title: Custom title for the plots (optional)
        make_pdf: If not False, save the cumulative distribution plot as a PDF with the given filename (optional)



    Returns:
        dict: Dictionary containing all calculated metrics
    """

    if save_dir != False:
        os.makedirs(save_dir, exist_ok=True)
    
    
    show_plots = True

    # Calculate errors
    relative_errors = np.abs(10**D_test - 10**D_pred) / (np.abs(10**D_test))
    relative_errors_flat = relative_errors.flatten()
    log_errors = np.abs(D_test - D_pred)
    log_errors_flat = log_errors.flatten()
    
    # Calculate statistics and store them in a dictionary
    metrics = {
        'linear': {
            'mean': np.mean(relative_errors_flat),
            'std': np.std(relative_errors_flat),
            'percentile_95': np.percentile(relative_errors_flat, 95),
             'percentile_50': np.percentile(relative_errors_flat, 50)
        },
        'log': {
            'mean': np.mean(log_errors_flat),
            'std': np.std(log_errors_flat),
            'percentile_95': np.percentile(log_errors_flat, 95),
            'percentile_50': np.percentile(log_errors_flat, 50)
        }
    }
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # # Create linear space plot
    # plt.figure(figsize=(6, 4))
    # plt.hist(relative_errors_flat, bins=np.logspace(-3, 1, 50), density=False, alpha=0.7)
    # plt.xscale('log')
    # plt.xlabel('Relative Error in Linear Space (log scale)')
    # plt.ylabel('Count')
    # title = (f'{name_of_test_prior} _ Distribution of Relative Errors Across All Gates\n'
    #         f'Mean: {metrics["linear"]["mean"]:.2%}, '
    #         f'Standard Deviation: {metrics["linear"]["std"]:.2%}\n'
    #         f'95th Percentile: {metrics["linear"]["percentile_95"]:.2%}')
    # if best_val_loss is not None:
    #     title += f'\nBest Validation Loss: {best_val_loss:.4e}'
    # else:
    #     print("No best validation loss provided.")
    # plt.title(title)
    
    # #Adds a lot of different lines
    # plt.axvline(metrics["linear"]["mean"], color='r', linestyle='dashed', 
    #             linewidth=1, label=f'Mean: {metrics["linear"]["mean"]:.2%}')
    # plt.axvline(metrics["linear"]["percentile_95"], color='b', linestyle='dashed',
    #             linewidth=1, label=f'95th Percentile: {metrics["linear"]["percentile_95"]:.2%}')
    # plt.axvline(metrics["linear"]["mean"] + metrics["linear"]["std"], color='g', linestyle='dashed',
    #             linewidth=1, label=f'Mean + Std: {(metrics["linear"]["mean"] + metrics["linear"]["std"]):.2%}')
    # plt.axvline(metrics["linear"]["percentile_50"], color='g', linestyle='dashed',
    #             linewidth=1, label=f'50th Percentile: {metrics["linear"]["percentile_50"]:.2%}')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()

    # # Save linear plot
    # filename = f'{name_of_test_prior}_relative_errors_histogram_linear_{timestamp}.png'
    # plt.savefig(os.path.join(save_dir, filename))
    # if show_plots:
    #     plt.show()
    # plt.close()
    
    # # Create log space plot
    # plt.figure(figsize=(6, 4))
    # plt.hist(log_errors_flat, bins=np.logspace(-3, 1, 50), density=False, alpha=0.7)
    # plt.xscale('log')
    # plt.xlabel('Absolute Error in Log Space (log scale)')
    # plt.ylabel('Count')
    # title = (f'{name_of_test_prior}_Distribution of Absolute Errors in Log Space Across All Gates\n'
    #         f'Mean: {metrics["log"]["mean"]:.2e}, '
    #         f'Standard Deviation: {metrics["log"]["std"]:.2e}\n'
    #         f'95th Percentile: {metrics["log"]["percentile_95"]:.2e}' +
    #         f'50th Percentile: {metrics["log"]["percentile_50"]:.2e}')
    
    # if best_val_loss is not None:
    #     title += f'\nBest Validation Loss: {best_val_loss:.4e}'
    # else:
    #     print("No best validation loss provided.")
    
    # plt.title(title)
    
    # plt.axvline(metrics["log"]["mean"], color='r', linestyle='dashed',
    #             linewidth=1, label=f'Mean: {metrics["log"]["mean"]:.2e}')
    # plt.axvline(metrics["log"]["percentile_95"], color='b', linestyle='dashed',
    #             linewidth=1, label=f'95th Percentile: {metrics["log"]["percentile_95"]:.2e}')
    # plt.axvline(metrics["log"]["mean"] + metrics["log"]["std"], color='g', linestyle='dashed',
    #             linewidth=1, label=f'Mean + Std: {(metrics["log"]["mean"] + metrics["log"]["std"]):.2e}')
    
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()

    # # Save log plot
    # filename = f'{name_of_test_prior}_absolute_errors_histogram_log_{timestamp}.png'
    # plt.savefig(os.path.join(save_dir, filename))
    # if show_plots:
    #     plt.show()
    # plt.close()
    
    # Create cumulative distribution plot
    plt.figure(figsize=(6, 4))
    
    # Sort errors and calculate cumulative percentages
    sorted_errors = np.sort(relative_errors_flat)
    # Calculate cumulative percentage
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    # Create the cumulative distribution plot
    plt.plot(sorted_errors, cumulative_pct, 'b-', linewidth=2, label='Cumulative distribution')
    #plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Relative error (%)')
    plt.ylabel('Cumulative percentage of predictions (%)')
    
    # Calculate percentage below threshold
    pct_below_threshold = (relative_errors_flat <= error_threshold).mean() * 100
    metrics['linear']['pct_below_threshold'] = pct_below_threshold
    
    #calculate for error threshold = 0.03
    pct_below_threshold_2 = (relative_errors_flat <= 0.03).mean() * 100
    metrics['linear']['pct_below_threshold_2'] = pct_below_threshold_2

    # Add reference lines
    plt.axvline(error_threshold, color='r', linestyle='--', 
                label=f'Error range 1: ±{error_threshold:.1%}')
    plt.axhline(pct_below_threshold, color='g', linestyle='--',
                label=f'{pct_below_threshold:.1f}% within error range 1')
    plt.axvline(0.03, color='k', linestyle='--',
                label=f'Error range 2: ±{0.03:.1%}')
    plt.axhline(metrics["linear"]["pct_below_threshold_2"], color='orange', linestyle='--',
                label=f'{metrics["linear"]["pct_below_threshold_2"]:.1f}% within error range 2')
    plt.axhline(95, color='b', linestyle='dashed',
                linewidth=1, label=f'95th percentile: {metrics["linear"]["percentile_95"]*100:.2f}%')

    # Add intersection point
    plt.plot(error_threshold, pct_below_threshold, 'ro')
    
    # Set axis limits
    plt.xlim(0,0.21)
    plt.xticks(np.arange(0,0.21, 0.03), [f"\u00B1{int(i*100)}%" for i in np.arange(0,0.21, 0.03)])  # Format as percentages
    plt.ylim(0, 100)
    
    
    title = title
    

    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save cumulative plot
    if save_dir:
        filename = f'{name_of_test_prior}_cumulative_relative_errors_{timestamp}.png'
        plt.savefig(os.path.join(save_dir, filename))

    if make_pdf != False:
        plt.savefig(make_pdf, bbox_inches='tight')  # Save as PDF with tight bounding box

    if show_plots:
        plt.show()
    plt.close()
    
    return metrics


import numpy as np
import matplotlib.pyplot as plt

#This function calculate the relative error for a specific gate and it is only used
#If i want a specific gate to be plotted.

def plot_relative_error_for_gate(D_test, D_pred, gate, noise_factor, save_dir='plots'):
    """
    Calculate and plot the relative error for a specific gate along with noise.

    Args:
        D_test: True values in log10 space
        D_pred: Predicted values in log10 space
        gate: The gate index to calculate the relative error for a specific gate
        noise_factor: Noise factor to add to the true values
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to linear space
    D_test_linear = 10**D_test
    D_pred_linear = 10**D_pred
    
    true_values_lin = D_test_linear[:, gate]
    pred_values_lin = D_pred_linear[:, gate]
    rel_error = (true_values_lin - pred_values_lin) / true_values_lin
    
    # Add noise
    noisy_values = np.random.normal(true_values_lin, noise_factor * true_values_lin)

    rel_error_noisy = (true_values_lin - noisy_values) / true_values_lin
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rel_error, bins=100, density=True, alpha=0.7, label='Error of NN')
    plt.hist(rel_error_noisy, bins=100, density=True, alpha=0.7, label='Error of noisy data')
    plt.xlim(-0.5, 0.5)
    plt.xlabel('Relative error')
    plt.ylabel('Density')
    plt.title(f'Gate {gate + 1}\n' +
              f'std: {np.std(rel_error):.2f}\n' +
              f'Noise factor: {noise_factor}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'relative_error_gate_{gate + 1}.png'))
    plt.show()
    


    #############################################
    #Function that calculated the std for every gate:
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_standard_deviation(D_true, D_pred, save_dir=None, show_plot=True, 
                           print_errors=True, fig_size=(6, 4), title=None,
                           filename='standard_deivation_per_gate.png'):
    """
    Analyze standard errors for each gate in neural network predictions.
    
    Parameters:
    -----------
    D_true : numpy array
        True values in log10 space
    D_pred : numpy array
        Predicted values in log10 space
    save_dir : str, optional
        Directory to save the plot
    show_plot : bool, optional
        Whether to display the plot
    print_errors : bool, optional
        Whether to print standard errors for each gate
    fig_size : tuple, optional
        Figure size for the plot
    title : str, optional
        Custom title for the plot
    filename : str, optional
        Filename for the saved plot
        
    Returns:
    --------
    standard_errors : list
        List of standard errors for each gate
    """
    # Convert to linear space
    D_true_linear = 10**D_true
    D_pred_linear = 10**D_pred

    # Calculate standard error for each gate
    standard_deviations = []
    for gate in range(D_true_linear.shape[1]):
        true_values = D_true_linear[:, gate]
        pred_values = D_pred_linear[:, gate]
        relative_error = (true_values - pred_values) / (true_values)
        standard_deviation = np.std(relative_error)
        standard_deviations.append(standard_deviation)

    # Plot standard errors for each gate
    plt.figure(figsize=fig_size)
    plt.bar(range(1, len(standard_deviations) + 1), standard_deviations, alpha=0.7)
    plt.xlabel('Gate')
    plt.ylabel('Standard deviation')
    plt.yscale('log')
    
    if title is None:
        plt.title('Standard deviation in Linear Space for Each Gate')
    else:
        plt.title(title)
        
    plt.grid(True)
    
    # Save plot if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Print standard errors for each gate if requested
    if print_errors:
        for gate, std_err in enumerate(standard_deviations, start=1):
            print(f"Gate {gate}: Standard Error = {std_err:.4e}")
    
    return standard_deviations



#Function made by Copiloit

def plot_percentage_below_threshold(D_true, D_pred, error_threshold=0.05, save_dir=None, 
                                     show_plot=True, fig_size=(6, 4), title=None, 
                                     filename='percentage_below_threshold_per_gate.png',
                                     error_threshold_2=0, make_pdf=False):
    """
    Plot the percentage of outputs below a relative error threshold for each gate as a line plot with dots.
    
    Parameters:
    -----------
    D_true : numpy array
        True values in log10 space
    D_pred : numpy array
        Predicted values in log10 space
    error_threshold : float, optional
        Relative error threshold (default: 0.05 for 5%)
    save_dir : str, optional
        Directory to save the plot
    show_plot : bool, optional
        Whether to display the plot
    fig_size : tuple, optional
        Figure size for the plot
    title : str, optional
        Custom title for the plot
    filename : str, optional
        Filename for the saved plot
        
    Returns:
    --------
    percentages_below_threshold : list
        List of percentages of outputs below the threshold for each gate
    """
    # Convert to linear space
    D_true_linear = 10**D_true
    D_pred_linear = 10**D_pred

    # Calculate the percentage of outputs below the threshold for each gate
    percentages_below_threshold = []
    for gate in range(D_true_linear.shape[1]):
        true_values = D_true_linear[:, gate]
        pred_values = D_pred_linear[:, gate]
        relative_error = np.abs(true_values - pred_values) / true_values
        percentage_below = np.mean(relative_error <= error_threshold) * 100
        percentages_below_threshold.append(percentage_below)

    if error_threshold_2 != 0:
        # Calculate the percentage of outputs below the second threshold for each gate
        percentages_below_threshold_2 = []
        for gate in range(D_true_linear.shape[1]):
            true_values = D_true_linear[:, gate]
            pred_values = D_pred_linear[:, gate]
            relative_error = np.abs(true_values - pred_values) / true_values
            percentage_below_2 = np.mean(relative_error <= error_threshold_2) * 100
            percentages_below_threshold_2.append(percentage_below_2)
        
    # Plot the percentages for each gate as a line plot with dots
    plt.figure(figsize=fig_size)
    plt.plot(range(1, len(percentages_below_threshold) + 1), percentages_below_threshold, 
             marker='o', linestyle='-', color='b', alpha=0.7, label=f'± {error_threshold:.1%} rel. error')
    plt.plot(range(1, len(percentages_below_threshold_2) + 1), percentages_below_threshold_2, 
             marker='o', linestyle='-', color='r', alpha=0.7, label=f'± {error_threshold_2:.1%} rel. error')
    plt.xlabel('Gate')
    plt.ylim(50, 100)
    plt.xlim(1, len(percentages_below_threshold))
    plt.ylabel(f'% outputs within error range (%)')
    
    if title is None:
        plt.title(f'Percentage of outputs below \u00B1{error_threshold:.1%} relative error for each gate')
    else:
        plt.title(title)
        
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    
    # Save plot if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
    
    if make_pdf != False:
        plt.savefig(make_pdf, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return percentages_below_threshold



#Function made by Copilot

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_percentage_and_std(D_true, D_pred, error_threshold=0.05, error_threshold_2=0, save_dir=None, 
                            show_plot=True, fig_size=(6, 4), title=None, 
                            filename='combined_plot_percentage_and_std.png', make_pdf=False):
    """
    Plot the percentage of outputs below relative error thresholds and standard deviations for each gate.

    Parameters:
    -----------
    D_true : numpy array
        True values in log10 space
    D_pred : numpy array
        Predicted values in log10 space
    error_threshold : float, optional
        Relative error threshold (default: 0.05 for 5%)
    error_threshold_2 : float, optional
        Second relative error threshold (default: 0)
    save_dir : str, optional
        Directory to save the plot
    show_plot : bool, optional
        Whether to display the plot
    fig_size : tuple, optional
        Figure size for the plot
    title : str, optional
        Custom title for the plot
    filename : str, optional
        Filename for the saved plot
    make_pdf : bool or str, optional
        If not False, save the plot as a PDF with the given filename

    Returns:
    --------
    None
    """
    # Convert to linear space
    D_true_linear = 10**D_true
    D_pred_linear = 10**D_pred

    # Calculate the percentage of outputs below the thresholds for each gate
    percentages_below_threshold = []
    percentages_below_threshold_2 = []
    standard_deviations = []

    for gate in range(D_true_linear.shape[1]):
        true_values = D_true_linear[:, gate]
        pred_values = D_pred_linear[:, gate]
        relative_error = np.abs(true_values - pred_values) / true_values 

        # Percentage below thresholds
        percentage_below = np.mean(relative_error <= error_threshold) * 100
        percentages_below_threshold.append(percentage_below)

        if error_threshold_2 != 0:
            percentage_below_2 = np.mean(relative_error <= error_threshold_2) * 100
            percentages_below_threshold_2.append(percentage_below_2)

        # Standard deviation
        relative_error_std = (true_values - pred_values) / true_values 
        standard_deviation = np.std(relative_error_std)
        standard_deviations.append(standard_deviation)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=fig_size)

    # Plot percentages below thresholds on the left y-axis
    ax1.plot(range(1, len(percentages_below_threshold) + 1), percentages_below_threshold, 
             marker='o', linestyle='-', color='b', alpha=0.7, label=f'±{error_threshold:.1%} rel. error')
    if error_threshold_2 != 0:
        ax1.plot(range(1, len(percentages_below_threshold_2) + 1), percentages_below_threshold_2, 
                 marker='o', linestyle='-', color='r', alpha=0.7, label=f' ±{error_threshold_2:.1%} rel. error')
    ax1.set_xlabel('Gate', fontsize=15)
    ax1.set_ylabel(f'% outputs within error range(%)', fontsize=15)
    ax1.set_ylim(50, 100)  # Set y-axis limits for the primary axis
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

   
    # Create a secondary y-axis for standard deviations
    ax2 = ax1.twinx()
    ax2.bar(range(1, len(standard_deviations) + 1), standard_deviations, color='gray', alpha=0.2, label='Std Dev')
    ax2.set_ylabel('Std Dev for relative errors', fontsize=15)
    ax2.set_ylim(10**(-2), 0.05)  # Set y-axis limits for the secondary axis
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_yscale('linear')  # Set y-axis to logarithmic scale




    # Combine legends into one
    handles1, labels1 = ax1.get_legend_handles_labels()  # Get elements from the primary y-axis
    handles2, labels2 = ax2.get_legend_handles_labels()  # Get elements from the secondary y-axis

    # Combine handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Add a single legend
    legend = ax2.legend(handles, labels, loc='lower right', fontsize=10, ncol=1)
    legend.get_frame().set_alpha(1)  # Fully opaque
    legend.get_frame().set_facecolor('white')  # White background
    legend.get_frame().set_edgecolor('black')  # Black border

    # Add a title
    if title is None:
        title = f'Percentage Below Thresholds and Standard Deviations for Each Gate'
    plt.title(title, fontsize=18)

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
    if make_pdf:
        plt.savefig(make_pdf, bbox_inches='tight')

    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return standard_deviations

