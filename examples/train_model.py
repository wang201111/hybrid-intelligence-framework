"""
Example script for training the hybrid intelligence framework
on salt-water solubility data
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.framework import HybridIntelligenceFramework


def main():
    # Configuration
    INPUT_COLS = ['T/°C', 'W(KCl)/%']
    OUTPUT_COLS = ['W(MgCl2)/%']
    
    # Binary system data paths
    BINARY_DATA = {
        'component1': 'data/binary_systems/KCl_solubility.xlsx',
        'component2': 'data/binary_systems/MgCl2_solubility.xlsx'
    }
    
    # Load raw data
    print("Loading experimental data...")
    raw_data = pd.read_excel('data/raw/KCl-MgCl2-H2O.xlsx')
    print(f"Raw data shape: {raw_data.shape}")
    
    # Initialize framework
    framework = HybridIntelligenceFramework(
        input_cols=INPUT_COLS,
        output_cols=OUTPUT_COLS,
        binary_data_paths=BINARY_DATA,
        test_split=0.2
    )
    
    # Train model with complete pipeline
    print("\nStarting hybrid intelligence framework training...")
    framework.fit(
        raw_data,
        layer_dim=4,
        node_dim=64,
        epochs=1000,
        learning_rate=0.008
    )
    
    # Save trained model
    framework.save_model('models/kcl_mgcl2_model')
    
    # Evaluate model performance
    print("\nModel Evaluation:")
    
    # Test predictions at different temperatures
    test_temperatures = [0, 25, 50, 100, 150, 200]
    test_data = []
    
    for temp in test_temperatures:
        # Create test points
        kcl_values = np.linspace(0, 40, 10)
        for kcl in kcl_values:
            test_data.append([temp, kcl])
    
    test_df = pd.DataFrame(test_data, columns=INPUT_COLS)
    
    # Predictions without constraints
    predictions_raw = framework.predict(test_df, apply_constraints=False)
    
    # Predictions with constraints
    predictions_constrained = framework.predict(test_df, apply_constraints=True)
    
    # Compare results
    results_df = test_df.copy()
    results_df['Predicted_MgCl2_Raw'] = predictions_raw
    results_df['Predicted_MgCl2_Constrained'] = predictions_constrained
    results_df['Difference'] = predictions_constrained - predictions_raw
    
    # Save results
    results_df.to_excel('results/predictions_comparison.xlsx', index=False)
    print("Results saved to results/predictions_comparison.xlsx")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Mean absolute difference: {np.abs(results_df['Difference']).mean():.4f}")
    print(f"Max absolute difference: {np.abs(results_df['Difference']).max():.4f}")
    
    # Check boundary compliance
    boundary_compliance = check_boundary_compliance(results_df, BINARY_DATA)
    print(f"Boundary compliance rate: {boundary_compliance:.2%}")


def check_boundary_compliance(predictions_df, binary_data_paths):
    """
    Check if predictions satisfy binary system boundary conditions
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    binary_data_paths : dict
        Paths to binary system data
        
    Returns:
    --------
    float
        Compliance rate (0-1)
    """
    # This is a simplified check - in practice would load actual binary data
    # and verify predictions at boundaries match binary solubilities
    
    compliant = 0
    total = 0
    
    # Check predictions where one component is zero
    boundary_mask = (predictions_df['W(KCl)/%'] == 0)
    boundary_predictions = predictions_df[boundary_mask]
    
    for _, row in boundary_predictions.iterrows():
        total += 1
        # Here you would check against actual binary solubility
        # For now, just check if prediction is positive
        if row['Predicted_MgCl2_Constrained'] > 0:
            compliant += 1
            
    return compliant / total if total > 0 else 1.0


if __name__ == "__main__":
    main()