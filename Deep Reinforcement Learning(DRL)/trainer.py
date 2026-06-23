"""Training script for autonomous driving model."""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from config import TRAINING_CONFIG, DATA_CONFIG
from model import create_model, save_model, load_model
from dataset import DataManager


class ModelTrainer:
    """Trainer class for autonomous driving model."""
    
    def __init__(self, device='cpu', model_name='autonomous_driving_model'):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.data_manager = DataManager()
    
    def initialize_model(self):
        """Initialize the model."""
        self.model = create_model(device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
        try:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
    
    def train_epoch(self, train_loader):
        """Train for one epoch with multimodal inputs."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sensors_tuple, labels) in enumerate(train_loader):
            # Unpack multimodal inputs
            camera, lidar, sensors = sensors_tuple
            
            camera = camera.to(self.device)
            lidar = lidar.to(self.device)
            sensors = sensors.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(camera, lidar, sensors)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model with multimodal inputs."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sensors_tuple, labels in val_loader:
                # Unpack multimodal inputs
                camera, lidar, sensors = sensors_tuple
                
                camera = camera.to(self.device)
                lidar = lidar.to(self.device)
                sensors = sensors.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(camera, lidar, sensors)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, epochs=TRAINING_CONFIG['epochs']):
        """Train the model for specified number of epochs."""
        print("Loading training data...")
        try:
            train_loader, val_loader = self.data_manager.create_data_loaders()
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        print(f"Starting training on device: {self.device}")
        self.initialize_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        print("\nTraining completed!")
        self._save_final_model()
        self.data_manager.save_scaler(os.path.join(DATA_CONFIG['models_dir'], 'scaler.pkl'))
    
    def _save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            DATA_CONFIG['models_dir'],
            f'{self.model_name}_epoch_{epoch+1}_loss_{val_loss:.4f}.pth'
        )
        save_model(self.model, checkpoint_path)
    
    def _save_final_model(self):
        """Save final model."""
        final_path = os.path.join(DATA_CONFIG['models_dir'], f'{self.model_name}_final.pth')
        save_model(self.model, final_path)


def train_model():
    """Main training function."""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer and train
    trainer = ModelTrainer(device=device)
    trainer.train(epochs=TRAINING_CONFIG['epochs'])


if __name__ == '__main__':
    train_model()
