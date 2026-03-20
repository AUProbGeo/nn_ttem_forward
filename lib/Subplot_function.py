#Subplot_function.py
import numpy as np
import matplotlib.pyplot as plt
import os

def create_subplots(D_test, D_pred, noise_factor, save_dir='subplots'
                    ,title='Relative Error Distribution of Neural Network Predictions'
                    ,savename='Test',make_pdf=False):
    """
    Create subplots, one for each gate, showing the relative error distribution.
    
    Args:
        D_test: True values in log10 space
        D_pred: Predicted values in log10 space
        noise_factor: Noise factor to add to the true values
        save_dir: Directory to save the plots
        title: Title of the figure default is 'Relative Error Distribution of Neural Network Predictions'
        savename: Name of the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to linear space
    D_test_linear = 10**D_test
    D_pred_linear = 10**D_pred
    
    # Create a figure with 40 subplots (5x8 grid)
    fig, axes = plt.subplots(8, 5, figsize=(30, 32))  # Reduce width from 40 to 30
    axes = axes.flatten()
   
    for gate in range(D_test_linear.shape[1]):
        true_values_lin = D_test_linear[:, gate]
        pred_values_lin = D_pred_linear[:, gate]
        rel_error = (true_values_lin - pred_values_lin) / (true_values_lin)
        
        # Makes data with noise and calculated rel. error.
        noisy_values=np.random.normal(true_values_lin, noise_factor*true_values_lin)
        rel_error_noisy = (true_values_lin - noisy_values) / (true_values_lin)
        
        # Define bins explicitly for the range -0.5 to 0.5
        bins = np.linspace(-0.4, 0.4, 101)  # 100 bins within the range
        

        ax = axes[gate]
        ax.hist(rel_error, bins=bins, density=True, alpha=0.8, color='blue', label='Relative error of predicted')  # Blue
        ax.hist(rel_error_noisy, bins=bins, density=True, alpha=0.6, color='orange', label='Noisy')  # Orange
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(0, 25)
        ax.set_xlabel('Relative Error', fontsize=24)  # Larger font for x-axis label
        ax.set_ylabel('Density', fontsize=24)    # Larger font for y-axis label
        ax.set_title(f'Gate {gate + 1}\nstd: {np.std(rel_error):.3f}', fontsize=22)  # Larger title font
        ax.tick_params(axis='both', labelsize=20)    # Larger tick labels
        ax.grid(True, linestyle='--', alpha=0.5)  # Subtle dashed gridlines for better readability
    
    
    # Add a single legend for the entire figure
    fig.legend([f'Model predictions',f'True values + gaussian noise \n Std Dev={noise_factor*100}%'], loc='upper right', ncol=1, fontsize=32, frameon=True, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.suptitle(title, fontsize=48, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    #plt.savefig(os.path.join(save_dir, f'{savename}_relative_error_subplots_noice_factor{noise_factor}.png')) # Save the figure
    if make_pdf != False:
        plt.savefig('Subplot_error_distribution.pdf', bbox_inches='tight')
    plt.show()
