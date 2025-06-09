"""
Low-Dimensional Physical Constraints (LDPC)
Implementation of physical constraint module using binary system boundary conditions
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess


class LOESSModel:
    """Local Weighted Regression (LOESS) for binary solubility data"""
    
    def __init__(self, frac=0.3):
        """
        Parameters:
        -----------
        frac : float
            Smoothing parameter controlling regression locality
        """
        self.frac = frac
        self.smoothed = None
        
    def fit(self, temperature, solubility):
        """
        Fit LOESS model to binary system data
        
        Parameters:
        -----------
        temperature : array-like
            Temperature values in °C
        solubility : array-like
            Solubility values in mass fraction
        """
        # Apply LOESS smoothing
        self.smoothed = lowess(solubility, temperature, frac=self.frac)
        return self
        
    def predict(self, temperature):
        """
        Predict solubility at given temperature using fitted model
        
        Parameters:
        -----------
        temperature : float or array-like
            Temperature value(s) for prediction
            
        Returns:
        --------
        float or array-like
            Predicted solubility value(s)
        """
        if self.smoothed is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Linear interpolation for prediction
        return np.interp(temperature, self.smoothed[:, 0], self.smoothed[:, 1])


class CurveTransformation:
    """
    Physics-based curve transformation with component-weighted mapping
    
    Transforms predicted solubility curves to satisfy binary system
    boundary conditions using weighted distance factors
    """
    
    @staticmethod
    def calculate_weight_factors(x1, x2, power=2):
        """
        Calculate weight factors based on component proportions
        
        Parameters:
        -----------
        x1, x2 : float
            Mass fractions of two components
        power : float
            Power for quadratic weighting function
            
        Returns:
        --------
        tuple
            Weight factors (α1, α2, α3) for three boundaries
        """
        # Normalize to ensure sum = 1
        total = x1 + x2
        if total == 0:
            return (1/3, 1/3, 1/3)
            
        x1_norm = x1 / total
        x2_norm = x2 / total
        
        # Calculate weights using quadratic function
        # w1: weight for x1=0 boundary (pure x2)
        w1 = (1 - x1_norm) ** power
        
        # w2: weight for x2=0 boundary (pure x1)  
        w2 = (1 - x2_norm) ** power
        
        # w3: weight for intermediate region
        w3 = 2 * x1_norm * x2_norm
        
        # Normalize weights
        w_sum = w1 + w2 + w3
        return (w1/w_sum, w2/w_sum, w3/w_sum)
    
    @staticmethod
    def transform_curve(curve, boundary_start, boundary_end):
        """
        Transform curve to satisfy boundary conditions
        
        Parameters:
        -----------
        curve : array-like, shape (n, 2)
            Original curve points (x, y)
        boundary_start : tuple
            Target start point (x, y)
        boundary_end : tuple
            Target end point (x, y)
            
        Returns:
        --------
        array-like, shape (n, 2)
            Transformed curve points
        """
        curve = np.array(curve)
        
        # Extract original endpoints
        x_min_idx = np.argmin(curve[:, 0])
        x_max_idx = np.argmax(curve[:, 0])
        original_start = curve[x_min_idx]
        original_end = curve[x_max_idx]
        
        # Convert boundary points to arrays
        target_start = np.array(boundary_start)
        target_end = np.array(boundary_end)
        
        # Step 1: Translation to align start points
        translation = target_start - original_start
        translated_curve = curve + translation
        
        # Step 2: Scaling to match boundary distance
        original_distance = np.linalg.norm(original_end - original_start)
        target_distance = np.linalg.norm(target_end - target_start)
        
        if original_distance > 0:
            scale_factor = target_distance / original_distance
        else:
            scale_factor = 1
            
        scaled_curve = (translated_curve - target_start) * scale_factor + target_start
        
        # Step 3: Rotation to align end points
        new_end = scaled_curve[x_max_idx]
        
        # Calculate rotation angle
        angle_target = np.arctan2(target_end[1] - target_start[1], 
                                 target_end[0] - target_start[0])
        angle_current = np.arctan2(new_end[1] - target_start[1], 
                                  new_end[0] - target_start[0])
        rotation_angle = angle_target - angle_current
        
        # Apply rotation matrix
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        
        # Rotate around start point
        centered_curve = scaled_curve - target_start
        rotated_curve = centered_curve @ rotation_matrix.T + target_start
        
        return rotated_curve


class LDPCConstraints:
    """
    Low-Dimensional Physical Constraints implementation
    
    Uses binary system solubility data as boundary conditions
    for ternary system predictions
    """
    
    def __init__(self, binary_data_paths=None):
        """
        Parameters:
        -----------
        binary_data_paths : dict
            Paths to binary system data files
            Expected keys: 'component1', 'component2'
        """
        self.binary_models = {}
        
        if binary_data_paths:
            self.load_binary_data(binary_data_paths)
            
    def load_binary_data(self, data_paths):
        """
        Load and fit LOESS models for binary systems
        
        Parameters:
        -----------
        data_paths : dict
            Dictionary with component names as keys and file paths as values
        """
        for component, path in data_paths.items():
            data = pd.read_excel(path)
            
            # Assume columns are 'T/°C' and 'Solubility'
            model = LOESSModel(frac=0.3)
            model.fit(data['T/°C'].values, data['Solubility'].values)
            
            self.binary_models[component] = model
            
    def apply(self, predictions, model, scaler_x, scaler_y, device):
        """
        Apply physical constraints to model predictions
        
        Parameters:
        -----------
        predictions : array-like, shape (n, 3)
            Predicted values [temperature, x1, x2]
        model : torch.nn.Module
            Trained neural network model
        scaler_x : sklearn scaler
            Input data scaler
        scaler_y : sklearn scaler
            Output data scaler
        device : torch.device
            Computing device
            
        Returns:
        --------
        array-like
            Constrained predictions
        """
        import torch
        
        constrained_predictions = predictions.copy()
        
        for i in range(len(predictions)):
            T = predictions[i, 0]
            x1 = predictions[i, 1]
            
            # Get binary solubilities at current temperature
            S1 = self.binary_models['component1'].predict(T)
            S2 = self.binary_models['component2'].predict(T)
            
            # Define boundary points
            boundary_1 = (S1, 0)  # Pure component 1
            boundary_2 = (0, S2)  # Pure component 2
            
            # Special cases: at boundaries
            if x1 == 0:
                constrained_predictions[i, 2] = S2
                continue
            elif x1 >= S1:
                constrained_predictions[i, 2] = 0
                continue
            
            # Generate prediction curve for current temperature
            x1_range = np.linspace(0, S1, 200)
            input_data = np.array([[T, x] for x in x1_range])
            
            # Scale and predict
            input_scaled = scaler_x.transform(input_data)
            input_tensor = torch.FloatTensor(input_scaled).to(device)
            
            model.eval()
            with torch.no_grad():
                output_scaled = model(input_tensor).cpu().numpy()
            
            output = scaler_y.inverse_transform(output_scaled)
            
            # Filter valid predictions
            curve_data = np.column_stack((x1_range, output.flatten()))
            valid_mask = curve_data[:, 1] > 0
            valid_curve = curve_data[valid_mask]
            
            if len(valid_curve) > 1:
                # Apply curve transformation
                transformer = CurveTransformation()
                transformed_curve = transformer.transform_curve(
                    valid_curve, boundary_2, boundary_1
                )
                
                # Interpolate to get constrained value
                if len(transformed_curve) > 0:
                    interp_func = interp1d(
                        transformed_curve[:, 0], 
                        transformed_curve[:, 1],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    
                    # Apply component-based weighting
                    direct_prediction = predictions[i, 2]
                    interpolated_value = interp_func(x1)
                    
                    # Calculate weight factors
                    weights = transformer.calculate_weight_factors(x1, predictions[i, 2])
                    
                    # Weighted combination of direct and transformed predictions
                    constrained_value = (
                        weights[0] * S2 +  # Boundary 1 contribution
                        weights[1] * S1 +  # Boundary 2 contribution  
                        weights[2] * interpolated_value  # Curve contribution
                    )
                    
                    # Ensure physical validity
                    constrained_predictions[i, 2] = max(0, min(constrained_value, S2))
                else:
                    constrained_predictions[i, 2] = 0
            else:
                constrained_predictions[i, 2] = 0
                
        return constrained_predictions