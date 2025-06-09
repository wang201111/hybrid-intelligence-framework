"""
Integrated Adaptive Data Augmentation Framework (IADAF)
Implementation of WGAN-GP based data augmentation with Bayesian optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


class Generator(nn.Module):
    """WGAN-GP Generator Network"""
    
    def __init__(self, latent_dim, hidden_dim, output_dim=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    """WGAN-GP Critic Network"""
    
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class IADAF:
    """
    Integrated Adaptive Data Augmentation Framework
    
    Combines WGAN-GP with Bayesian optimization to automatically
    adjust hyperparameters for different chemical systems
    """
    
    def __init__(self, latent_dim_range=(10, 256), hidden_dim_range=(10, 512),
                 n_critic_range=(1, 20), lambda_gp_range=(0.1, 10),
                 n_samples=1500, device=None):
        """
        Parameters:
        -----------
        latent_dim_range : tuple
            Range for latent dimension in Bayesian optimization
        hidden_dim_range : tuple
            Range for hidden layer dimension
        n_critic_range : tuple
            Range for critic iterations per generator iteration
        lambda_gp_range : tuple
            Range for gradient penalty coefficient
        n_samples : int
            Number of synthetic samples to generate
        device : torch.device
            Computing device (CPU/GPU)
        """
        self.latent_dim_range = latent_dim_range
        self.hidden_dim_range = hidden_dim_range
        self.n_critic_range = n_critic_range
        self.lambda_gp_range = lambda_gp_range
        self.n_samples = n_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimization results storage
        self.best_params = None
        self.generator = None
        
    def _gradient_penalty(self, critic, real_data, fake_data, lambda_gp):
        """
        Calculate gradient penalty for WGAN-GP
        
        Enforces 1-Lipschitz constraint on the critic
        """
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1).to(self.device)
        epsilon = epsilon.expand_as(real_data)

        # Interpolate between real and fake data
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        # Calculate critic scores
        interpolated_score = critic(interpolated)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=interpolated_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return penalty
    
    def _train_wgan_gp(self, data, latent_dim, hidden_dim, n_critic, lambda_gp,
                       epochs=2000, batch_size=64):
        """
        Train WGAN-GP with specified hyperparameters
        
        Returns the trained generator and training loss
        """
        # Data normalization
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(data)
        X_train = torch.FloatTensor(X_train).to(self.device)
        
        # Initialize networks
        generator = Generator(int(latent_dim), int(hidden_dim), data.shape[1]).to(self.device)
        critic = Critic(data.shape[1], int(hidden_dim)).to(self.device)
        
        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        # Training loop
        n_critic = int(n_critic)
        for epoch in range(epochs):
            # Train critic
            for _ in range(n_critic):
                # Sample real and fake data
                idx = torch.randint(0, len(X_train), (batch_size,))
                real_data = X_train[idx]
                
                noise = torch.randn(batch_size, int(latent_dim)).to(self.device)
                fake_data = generator(noise)
                
                # Critic loss
                real_score = critic(real_data)
                fake_score = critic(fake_data.detach())
                gp = self._gradient_penalty(critic, real_data, fake_data, lambda_gp)
                
                d_loss = fake_score.mean() - real_score.mean() + gp
                
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
            
            # Train generator
            noise = torch.randn(batch_size, int(latent_dim)).to(self.device)
            fake_data = generator(noise)
            g_loss = -critic(fake_data).mean()
            
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
        
        # Store scaler for later use
        self.scaler = scaler
        
        return generator, g_loss.item()
    
    def _quality_filter(self, original_data, generated_data):
        """
        Filter generated data using XGBoost-based quality assessment
        
        Removes synthetic samples with high prediction errors
        """
        # Combine data
        original_df = pd.DataFrame(original_data)
        original_df['Type'] = 'Original'
        
        generated_df = pd.DataFrame(generated_data)
        generated_df['Type'] = 'Generated'
        
        combined_df = pd.concat([original_df, generated_df], ignore_index=True)
        
        # Prepare features and targets
        feature_cols = list(range(len(original_data.columns) - 1))
        target_col = len(original_data.columns) - 1
        
        X = combined_df.iloc[:, feature_cols]
        y = combined_df.iloc[:, target_col]
        
        # Scale data
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=7,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        model.fit(X_scaled, y_scaled)
        
        # Predict and calculate errors
        predictions_scaled = model.predict(X_scaled)
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        errors = np.abs(predictions - y.values)
        
        # Filter based on error threshold
        error_threshold = 15  # Adjustable based on system requirements
        mask = errors <= error_threshold
        filtered_data = combined_df[mask]
        
        return filtered_data[filtered_data['Type'] == 'Generated'].drop(columns=['Type'])
    
    def _objective_function(self, data, **params):
        """
        Objective function for Bayesian optimization
        
        Trains WGAN-GP and returns negative loss as optimization target
        """
        generator, loss = self._train_wgan_gp(
            data,
            latent_dim=params['latent_dim'],
            hidden_dim=params['hidden_dim'],
            n_critic=params['n_critic'],
            lambda_gp=params['lambda_gp']
        )
        
        return -loss  # Maximize by minimizing negative loss
    
    def fit_generate(self, data):
        """
        Fit the augmentation framework and generate synthetic data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original training data
            
        Returns:
        --------
        pd.DataFrame
            Combined original and synthetic data
        """
        # Define parameter bounds for optimization
        pbounds = {
            'latent_dim': self.latent_dim_range,
            'hidden_dim': self.hidden_dim_range,
            'n_critic': self.n_critic_range,
            'lambda_gp': self.lambda_gp_range
        }
        
        # Bayesian optimization
        optimizer = BayesianOptimization(
            f=lambda **params: self._objective_function(data.values, **params),
            pbounds=pbounds,
            random_state=42
        )
        
        # Run optimization
        optimizer.maximize(init_points=5, n_iter=15)
        
        # Extract best parameters
        self.best_params = optimizer.max['params']
        print(f"Optimal hyperparameters: {self.best_params}")
        
        # Train final model with best parameters
        self.generator, _ = self._train_wgan_gp(
            data.values,
            **self.best_params
        )
        
        # Generate synthetic samples
        noise = torch.randn(self.n_samples, int(self.best_params['latent_dim'])).to(self.device)
        generated_data = self.generator(noise).cpu().detach().numpy()
        
        # Denormalize
        generated_data = self.scaler.inverse_transform(generated_data)
        generated_df = pd.DataFrame(generated_data, columns=data.columns)
        
        # Quality filtering
        filtered_generated = self._quality_filter(data, generated_df)
        
        # Combine with original data
        augmented_data = pd.concat([data, filtered_generated], ignore_index=True)
        
        print(f"Generated {len(filtered_generated)} high-quality synthetic samples")
        print(f"Total dataset size: {len(augmented_data)}")
        
        return augmented_data