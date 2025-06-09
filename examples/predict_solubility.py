"""
Example script for making solubility predictions using
a trained hybrid intelligence framework model
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from src.framework import HybridIntelligenceFramework


def predict_solubility_curve(framework, temperature, kcl_range=(0, 50)):
    """
    Generate solubility curve at a specific temperature
    
    Parameters:
    -----------
    framework : HybridIntelligenceFramework
        Trained framework instance
    temperature : float
        Temperature in °C
    kcl_range : tuple
        Range of KCl mass fraction values
        
    Returns:
    --------
    pd.DataFrame
        Predictions with and without constraints
    """
    # Generate input points
    kcl_values = np.linspace(kcl_range[0], kcl_range[1], 100)
    input_data = pd.DataFrame({
        'T/°C': [temperature] * len(kcl_values),
        'W(KCl)/%': kcl_values
    })
    
    # Make predictions
    predictions_raw = framework.predict(input_data, apply_constraints=False)
    predictions_constrained = framework.predict(input_data, apply_constraints=True)
    
    # Combine results
    results = input_data.copy()
    results['W(MgCl2)/%_raw'] = predictions_raw
    results['W(MgCl2)/%_constrained'] = predictions_constrained
    
    return results


def predict_temperature_series(framework, kcl_value, temp_range=(-30, 250)):
    """
    Generate predictions across temperature range for fixed KCl concentration
    
    Parameters:
    -----------
    framework : HybridIntelligenceFramework
        Trained framework instance
    kcl_value : float
        Fixed KCl mass fraction
    temp_range : tuple
        Temperature range in °C
        
    Returns:
    --------
    pd.DataFrame
        Predictions across temperatures
    """
    # Generate temperature points
    temperatures = np.linspace(temp_range[0], temp_range[1], 50)
    input_data = pd.DataFrame({
        'T/°C': temperatures,
        'W(KCl)/%': [kcl_value] * len(temperatures)
    })
    
    # Make predictions
    predictions_raw = framework.predict(input_data, apply_constraints=False)
    predictions_constrained = framework.predict(input_data, apply_constraints=True)
    
    # Combine results
    results = input_data.copy()
    results['W(MgCl2)/%_raw'] = predictions_raw
    results['W(MgCl2)/%_constrained'] = predictions_constrained
    
    return results


def main():
    # Initialize framework
    framework = HybridIntelligenceFramework(
        input_cols=['T/°C', 'W(KCl)/%'],
        output_cols=['W(MgCl2)/%']
    )
    
    # Load trained model
    print("Loading trained model...")
    framework.load_model('models/kcl_mgcl2_model')
    
    # Example 1: Predict solubility curve at specific temperatures
    print("\nExample 1: Solubility curves at different temperatures")
    temperatures = [0, 25, 50, 100, 150, 200]
    
    all_curves = []
    for temp in temperatures:
        print(f"  Generating curve at {temp}°C...")
        curve = predict_solubility_curve(framework, temp)
        curve['Temperature'] = temp
        all_curves.append(curve)
    
    # Save curves
    curves_df = pd.concat(all_curves, ignore_index=True)
    curves_df.to_excel('results/solubility_curves.xlsx', index=False)
    print("  Saved to results/solubility_curves.xlsx")
    
    # Example 2: Temperature series for fixed compositions
    print("\nExample 2: Temperature dependence at fixed KCl concentrations")
    kcl_values = [5, 10, 15, 20, 25, 30]
    
    all_series = []
    for kcl in kcl_values:
        print(f"  Generating series for W(KCl)={kcl}%...")
        series = predict_temperature_series(framework, kcl)
        series['KCl_concentration'] = kcl
        all_series.append(series)
    
    # Save series
    series_df = pd.concat(all_series, ignore_index=True)
    series_df.to_excel('results/temperature_series.xlsx', index=False)
    print("  Saved to results/temperature_series.xlsx")
    
    # Example 3: Single point predictions
    print("\nExample 3: Single point predictions")
    test_points = [
        {'T/°C': 25, 'W(KCl)/%': 10},
        {'T/°C': 50, 'W(KCl)/%': 15},
        {'T/°C': 100, 'W(KCl)/%': 20},
        {'T/°C': 150, 'W(KCl)/%': 25},
        {'T/°C': 200, 'W(KCl)/%': 30}
    ]
    
    test_df = pd.DataFrame(test_points)
    predictions_raw = framework.predict(test_df, apply_constraints=False)
    predictions_constrained = framework.predict(test_df, apply_constraints=True)
    
    # Display results
    print("\n  Temperature  W(KCl)/%  W(MgCl2)/% (Raw)  W(MgCl2)/% (Constrained)")
    print("  " + "-" * 65)
    for i, row in test_df.iterrows():
        print(f"  {row['T/°C']:>11.1f}  {row['W(KCl)/%']:>8.1f}  "
              f"{predictions_raw[i, 0]:>16.2f}  {predictions_constrained[i, 0]:>23.2f}")
    
    # Example 4: Extrapolation capability test
    print("\nExample 4: Testing extrapolation capability")
    
    # High temperature extrapolation
    high_temp_data = pd.DataFrame({
        'T/°C': [220, 240, 260, 280, 300],
        'W(KCl)/%': [15, 15, 15, 15, 15]
    })
    
    high_temp_pred = framework.predict(high_temp_data, apply_constraints=True)
    
    print("\n  High temperature extrapolation (W(KCl)=15%):")
    print("  Temperature  W(MgCl2)/% Predicted")
    print("  " + "-" * 35)
    for i, temp in enumerate(high_temp_data['T/°C']):
        print(f"  {temp:>11.0f}  {high_temp_pred[i, 0]:>19.2f}")
    
    # Save all results summary
    summary = {
        'Test Points': test_df.to_dict(),
        'Raw Predictions': predictions_raw.tolist(),
        'Constrained Predictions': predictions_constrained.tolist(),
        'High Temp Extrapolation': {
            'Temperatures': high_temp_data['T/°C'].tolist(),
            'Predictions': high_temp_pred.tolist()
        }
    }
    
    import json
    with open('results/prediction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll predictions completed successfully!")


if __name__ == "__main__":
    main()