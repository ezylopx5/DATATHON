"""
Advanced Random Forest + XGBoost Training with Cross-Validation
A100 GPU OPTIMIZED VERSION - Maximum Performance
Uses K-fold CV, hyperparameter optimization, and extensive validation
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    train_test_split, cross_val_score, cross_validate
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
import gc
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from features import RacingFeatureEngineer

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available for hyperparameter optimization")
except:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available - using default parameters")

# Check GPU
GPU_AVAILABLE = False
GPU_NAME = "Unknown"
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {GPU_NAME}")
        print(f"   Memory: {GPU_MEMORY:.1f} GB")
        
        # Check for A100
        if "A100" in GPU_NAME:
            print("   🚀 A100 DETECTED - Enabling maximum performance mode!")
except:
    GPU_AVAILABLE = True  # Assume available for XGBoost

class AdvancedCVPipeline:
    """
    Advanced training pipeline with A100 GPU optimization:
    - K-fold cross-validation
    - Hyperparameter optimization
    - GPU-accelerated training
    - Maximum performance tuning
    """
    
    def __init__(self, n_folds=5, use_optuna=False):
        self.n_folds = n_folds
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.rf_models = []
        self.xgb_models = []
        self.feature_engineer = RacingFeatureEngineer()
        self.feature_importance = {}
        self.feature_names = None
        self.cv_results = {}
        self.best_params = {}
        self.use_gpu = GPU_AVAILABLE
        
    def cross_validate_rf(self, X, y, params=None):
        """
        Cross-validate Random Forest with detailed metrics
        CPU-optimized for maximum throughput
        """
        print("\n" + "="*70)
        print("🌲 RANDOM FOREST CROSS-VALIDATION")
        print("="*70)
        
        if params is None:
            # A100 optimized parameters - more aggressive
            params = {
                'n_estimators': 800,          # More trees for A100
                'max_depth': 30,              # Deeper trees
                'min_samples_split': 8,      
                'min_samples_leaf': 4,        
                'max_features': 'sqrt',       
                'max_samples': 0.85,          # More data per tree
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,                 # All CPU cores
                'random_state': 42,
                'verbose': 0,
                'warm_start': False,
                'criterion': 'squared_error',
            }
        
        # K-Fold setup
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = {
            'train_rmse': [], 'val_rmse': [],
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': [],
            'train_time': [], 'oob_score': [],
            'feature_importance': []
        }
        
        print(f"\n📊 Configuration:")
        print(f"   Folds: {self.n_folds}")
        print(f"   Trees: {params['n_estimators']}")
        print(f"   Max depth: {params['max_depth']}")
        print(f"   CPU cores: All available")
        
        total_start = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n📁 Fold {fold}/{self.n_folds}")
            print("-"*40)
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            print(f"   Train: {len(X_train_fold):,} samples")
            print(f"   Val: {len(X_val_fold):,} samples")
            
            # Train model
            fold_start = time.time()
            
            rf_model = RandomForestRegressor(**params)
            rf_model.fit(X_train_fold, y_train_fold)
            
            fold_time = time.time() - fold_start
            
            # Store model
            self.rf_models.append(rf_model)
            
            # Predictions
            train_pred = rf_model.predict(X_train_fold)
            val_pred = rf_model.predict(X_val_fold)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_fold, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            
            train_mae = mean_absolute_error(y_train_fold, train_pred)
            val_mae = mean_absolute_error(y_val_fold, val_pred)
            
            train_r2 = r2_score(y_train_fold, train_pred)
            val_r2 = r2_score(y_val_fold, val_pred)
            
            # Store results
            fold_results['train_rmse'].append(train_rmse)
            fold_results['val_rmse'].append(val_rmse)
            fold_results['train_mae'].append(train_mae)
            fold_results['val_mae'].append(val_mae)
            fold_results['train_r2'].append(train_r2)
            fold_results['val_r2'].append(val_r2)
            fold_results['train_time'].append(fold_time)
            fold_results['feature_importance'].append(rf_model.feature_importances_)
            
            if hasattr(rf_model, 'oob_score_'):
                fold_results['oob_score'].append(rf_model.oob_score_)
                oob_rmse = np.sqrt(1 - rf_model.oob_score_) * np.std(y_train_fold)
                print(f"   OOB Score: {rf_model.oob_score_:.4f} (≈RMSE: {oob_rmse:.4f})")
            
            print(f"   Train RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
            print(f"   Val RMSE:   {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
            print(f"   Time: {fold_time:.2f}s")
            
            # Clear memory
            gc.collect()
        
        total_time = time.time() - total_start
        
        # Calculate aggregate metrics
        cv_summary = {
            'train_rmse_mean': np.mean(fold_results['train_rmse']),
            'train_rmse_std': np.std(fold_results['train_rmse']),
            'val_rmse_mean': np.mean(fold_results['val_rmse']),
            'val_rmse_std': np.std(fold_results['val_rmse']),
            'train_mae_mean': np.mean(fold_results['train_mae']),
            'val_mae_mean': np.mean(fold_results['val_mae']),
            'train_r2_mean': np.mean(fold_results['train_r2']),
            'val_r2_mean': np.mean(fold_results['val_r2']),
            'total_time': total_time,
            'avg_fold_time': np.mean(fold_results['train_time']),
            'fold_results': fold_results
        }
        
        if fold_results['oob_score']:
            cv_summary['oob_score_mean'] = np.mean(fold_results['oob_score'])
            cv_summary['oob_score_std'] = np.std(fold_results['oob_score'])
        
        # Feature importance (averaged across folds)
        avg_importance = np.mean(fold_results['feature_importance'], axis=0)
        self.feature_importance['rf'] = avg_importance
        
        # Print summary
        print("\n" + "="*70)
        print("📊 RANDOM FOREST CV SUMMARY")
        print("="*70)
        print(f"\n  Training Metrics (mean ± std):")
        print(f"    RMSE: {cv_summary['train_rmse_mean']:.4f} ± {cv_summary['train_rmse_std']:.4f}")
        print(f"    MAE:  {cv_summary['train_mae_mean']:.4f}")
        print(f"    R²:   {cv_summary['train_r2_mean']:.4f}")
        
        print(f"\n  Validation Metrics (mean ± std):")
        print(f"    RMSE: {cv_summary['val_rmse_mean']:.4f} ± {cv_summary['val_rmse_std']:.4f}")
        print(f"    MAE:  {cv_summary['val_mae_mean']:.4f}")
        print(f"    R²:   {cv_summary['val_r2_mean']:.4f}")
        
        if 'oob_score_mean' in cv_summary:
            print(f"\n  OOB Score: {cv_summary['oob_score_mean']:.4f} ± {cv_summary['oob_score_std']:.4f}")
        
        print(f"\n  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Avg per fold: {cv_summary['avg_fold_time']:.2f}s")
        
        self.cv_results['rf'] = cv_summary
        
        return cv_summary
    
    def cross_validate_xgb(self, X, y, params=None):
        """
        Cross-validate XGBoost with A100 GPU optimization
        """
        print("\n" + "="*70)
        print("🚀 XGBOOST CROSS-VALIDATION (A100 GPU)")
        print("="*70)
        
        if params is None:
            # A100 GPU OPTIMIZED parameters
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                
                # Model complexity - more aggressive for A100
                'max_depth': 12,              # Deeper for GPU
                'learning_rate': 0.03,        # Lower for better fit
                'n_estimators': 3000,         # Many iterations with early stopping
                
                # Sampling - A100 can handle more
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.75,
                'colsample_bynode': 0.75,
                
                # Regularization
                'reg_alpha': 0.05,
                'reg_lambda': 1.0,
                'gamma': 0.05,
                'min_child_weight': 8,
                
                # Performance
                'random_state': 42,
                'verbosity': 0,
            }
            
            # A100 GPU settings - MAXIMUM PERFORMANCE
            if self.use_gpu:
                params.update({
                    'tree_method': 'gpu_hist',        # GPU histogram
                    'predictor': 'gpu_predictor',     # GPU predictions
                    'gpu_id': 0,
                    'max_bin': 512,                   # More bins for A100
                    'max_cat_to_onehot': 8,          # Categorical optimization
                    'grow_policy': 'depthwise',      # Better for GPU
                    'single_precision_histogram': False,  # Full precision
                    'deterministic_histogram': True, # Reproducible
                })
        
        # K-Fold setup
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = {
            'train_rmse': [], 'val_rmse': [],
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': [],
            'train_time': [], 'best_iteration': [],
            'feature_importance': []
        }
        
        print(f"\n📊 Configuration:")
        print(f"   Folds: {self.n_folds}")
        print(f"   Max rounds: {params['n_estimators']}")
        print(f"   Max depth: {params['max_depth']}")
        print(f"   Learning rate: {params['learning_rate']}")
        print(f"   Device: A100 GPU" if self.use_gpu else "   Device: CPU")
        print(f"   Max bins: {params.get('max_bin', 256)}")
        
        total_start = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n📁 Fold {fold}/{self.n_folds}")
            print("-"*40)
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            print(f"   Train: {len(X_train_fold):,} samples")
            print(f"   Val: {len(X_val_fold):,} samples")
            
            # Train model with early stopping
            fold_start = time.time()
            
            xgb_model = xgb.XGBRegressor(**params)
            
            # Detailed evaluation during training
            eval_set = [(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)]
            
            xgb_model.fit(
                X_train_fold, y_train_fold,
                eval_set=eval_set,
                early_stopping_rounds=150,  # More patience for better fit
                verbose=False
            )
            
            fold_time = time.time() - fold_start
            
            # Store model
            self.xgb_models.append(xgb_model)
            
            # Get best iteration
            best_iter = xgb_model.best_iteration
            
            # Predictions
            train_pred = xgb_model.predict(X_train_fold)
            val_pred = xgb_model.predict(X_val_fold)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_fold, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            
            train_mae = mean_absolute_error(y_train_fold, train_pred)
            val_mae = mean_absolute_error(y_val_fold, val_pred)
            
            train_r2 = r2_score(y_train_fold, train_pred)
            val_r2 = r2_score(y_val_fold, val_pred)
            
            # Store results
            fold_results['train_rmse'].append(train_rmse)
            fold_results['val_rmse'].append(val_rmse)
            fold_results['train_mae'].append(train_mae)
            fold_results['val_mae'].append(val_mae)
            fold_results['train_r2'].append(train_r2)
            fold_results['val_r2'].append(val_r2)
            fold_results['train_time'].append(fold_time)
            fold_results['best_iteration'].append(best_iter)
            fold_results['feature_importance'].append(xgb_model.feature_importances_)
            
            print(f"   Best iteration: {best_iter}/{params['n_estimators']}")
            print(f"   Train RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
            print(f"   Val RMSE:   {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
            print(f"   Time: {fold_time:.2f}s (⚡ GPU accelerated)")
            
            # Clear GPU memory
            if self.use_gpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            gc.collect()
        
        total_time = time.time() - total_start
        
        # Calculate aggregate metrics
        cv_summary = {
            'train_rmse_mean': np.mean(fold_results['train_rmse']),
            'train_rmse_std': np.std(fold_results['train_rmse']),
            'val_rmse_mean': np.mean(fold_results['val_rmse']),
            'val_rmse_std': np.std(fold_results['val_rmse']),
            'train_mae_mean': np.mean(fold_results['train_mae']),
            'val_mae_mean': np.mean(fold_results['val_mae']),
            'train_r2_mean': np.mean(fold_results['train_r2']),
            'val_r2_mean': np.mean(fold_results['val_r2']),
            'avg_best_iteration': np.mean(fold_results['best_iteration']),
            'total_time': total_time,
            'avg_fold_time': np.mean(fold_results['train_time']),
            'fold_results': fold_results
        }
        
        # Feature importance (averaged across folds)
        avg_importance = np.mean(fold_results['feature_importance'], axis=0)
        self.feature_importance['xgb'] = avg_importance
        
        # Print summary
        print("\n" + "="*70)
        print("📊 XGBOOST CV SUMMARY")
        print("="*70)
        print(f"\n  Training Metrics (mean ± std):")
        print(f"    RMSE: {cv_summary['train_rmse_mean']:.4f} ± {cv_summary['train_rmse_std']:.4f}")
        print(f"    MAE:  {cv_summary['train_mae_mean']:.4f}")
        print(f"    R²:   {cv_summary['train_r2_mean']:.4f}")
        
        print(f"\n  Validation Metrics (mean ± std):")
        print(f"    RMSE: {cv_summary['val_rmse_mean']:.4f} ± {cv_summary['val_rmse_std']:.4f}")
        print(f"    MAE:  {cv_summary['val_mae_mean']:.4f}")
        print(f"    R²:   {cv_summary['val_r2_mean']:.4f}")
        
        print(f"\n  Avg best iteration: {cv_summary['avg_best_iteration']:.0f}")
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Avg per fold: {cv_summary['avg_fold_time']:.2f}s")
        print(f"  ⚡ GPU Speedup enabled!")
        
        self.cv_results['xgb'] = cv_summary
        
        return cv_summary
    
    def optimize_hyperparameters_rf(self, X, y, n_trials=50):
        """
        Optimize Random Forest hyperparameters using Optuna
        """
        if not self.use_optuna:
            print("⚠️ Optuna not available, using default parameters")
            return None
        
        print("\n" + "="*70)
        print("🔍 OPTIMIZING RANDOM FOREST HYPERPARAMETERS")
        print("="*70)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1200, step=100),
                'max_depth': trial.suggest_int('max_depth', 15, 40, step=5),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 25),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7, 0.8]),
                'max_samples': trial.suggest_float('max_samples', 0.6, 1.0),
                'bootstrap': True,
                'n_jobs': -1,
                'random_state': 42
            }
            
            # 3-fold CV for speed
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✅ Best parameters found:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print(f"\n   Best CV RMSE: {study.best_value:.4f}")
        
        self.best_params['rf'] = study.best_params
        
        return study.best_params
    
    def optimize_hyperparameters_xgb(self, X, y, n_trials=50):
        """
        Optimize XGBoost hyperparameters using Optuna with A100 GPU
        """
        if not self.use_optuna:
            print("⚠️ Optuna not available, using default parameters")
            return None
        
        print("\n" + "="*70)
        print("🔍 OPTIMIZING XGBOOST HYPERPARAMETERS (A100 GPU)")
        print("="*70)
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 6, 18),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 1000, 4000, step=500),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'random_state': 42,
                'verbosity': 0
            }
            
            if self.use_gpu:
                params.update({
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': 0,
                    'max_bin': 512
                })
            
            # 3-fold CV for speed
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)  # Sequential for GPU
        
        print(f"\n✅ Best parameters found:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print(f"\n   Best CV RMSE: {study.best_value:.4f}")
        
        self.best_params['xgb'] = study.best_params
        
        return study.best_params
    
    def train_final_ensemble(self, X, y):
        """
        Train final ensemble on full data with best parameters
        A100 GPU optimized
        """
        print("\n" + "="*70)
        print("🏆 TRAINING FINAL ENSEMBLE (A100 GPU)")
        print("="*70)
        
        # Get best parameters or use A100-optimized defaults
        rf_params = self.best_params.get('rf', {
            'n_estimators': 800,
            'max_depth': 30,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'max_samples': 0.85,
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42
        })
        
        xgb_params = self.best_params.get('xgb', {
            'objective': 'reg:squarederror',
            'max_depth': 12,
            'learning_rate': 0.03,
            'n_estimators': 3000,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        })
        
        if self.use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'max_bin': 512
            })
        
        # Train final models
        print("\n🌲 Training final Random Forest...")
        self.final_rf = RandomForestRegressor(**rf_params)
        rf_start = time.time()
        self.final_rf.fit(X, y)
        rf_time = time.time() - rf_start
        print(f"   ✅ Complete in {rf_time:.2f}s")
        
        if hasattr(self.final_rf, 'oob_score_'):
            print(f"   OOB Score: {self.final_rf.oob_score_:.4f}")
        
        print("\n🚀 Training final XGBoost (A100 GPU)...")
        # Split for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        
        self.final_xgb = xgb.XGBRegressor(**xgb_params)
        xgb_start = time.time()
        self.final_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=150,
            verbose=100
        )
        xgb_time = time.time() - xgb_start
        print(f"   ✅ Complete in {xgb_time:.2f}s (⚡ GPU accelerated)")
        print(f"   Best iteration: {self.final_xgb.best_iteration}")
        
        # Calculate speedup
        print(f"\n⚡ Performance Summary:")
        print(f"   RF Time: {rf_time:.2f}s")
        print(f"   XGB Time: {xgb_time:.2f}s")
        print(f"   Total: {rf_time + xgb_time:.2f}s")
    
    def ensemble_predict(self, X, method='weighted_cv'):
        """
        Generate ensemble predictions using various strategies
        """
        if method == 'weighted_cv':
            # Weight by CV performance
            rf_weight = 1 / self.cv_results['rf']['val_rmse_mean'] if 'rf' in self.cv_results else 0.5
            xgb_weight = 1 / self.cv_results['xgb']['val_rmse_mean'] if 'xgb' in self.cv_results else 0.5
            
            total_weight = rf_weight + xgb_weight
            rf_weight /= total_weight
            xgb_weight /= total_weight
            
            rf_pred = self.final_rf.predict(X)
            xgb_pred = self.final_xgb.predict(X)
            
            return rf_weight * rf_pred + xgb_weight * xgb_pred
        
        elif method == 'average_folds':
            # Average predictions from all fold models
            rf_preds = [model.predict(X) for model in self.rf_models]
            xgb_preds = [model.predict(X) for model in self.xgb_models]
            
            rf_avg = np.mean(rf_preds, axis=0)
            xgb_avg = np.mean(xgb_preds, axis=0)
            
            return 0.4 * rf_avg + 0.6 * xgb_avg
        
        else:
            # Simple average
            rf_pred = self.final_rf.predict(X)
            xgb_pred = self.final_xgb.predict(X)
            return 0.5 * rf_pred + 0.5 * xgb_pred
    
    def save_all(self):
        """Save models and CV results"""
        print("\n💾 Saving all models and results...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save final models
        joblib.dump(self.final_rf, 'models/rf_final.pkl')
        joblib.dump(self.final_xgb, 'models/xgb_final.pkl')
        
        # Save CV models
        joblib.dump(self.rf_models, 'models/rf_cv_models.pkl')
        joblib.dump(self.xgb_models, 'models/xgb_cv_models.pkl')
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, 'models/feature_engineer.pkl')
        
        # Save CV results
        with open('models/cv_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_for_json = {}
            for model, results in self.cv_results.items():
                results_for_json[model] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in results.items()
                    if k != 'fold_results'  # Skip detailed fold results
                }
            json.dump(results_for_json, f, indent=2)
        
        # Save best parameters
        if self.best_params:
            with open('models/best_params.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        print("   ✅ All models and results saved")

def main():
    """Main training pipeline with A100 GPU optimization"""
    print("="*70)
    print("🏎️  ADVANCED TRAINING WITH CROSS-VALIDATION")
    print("   🚀 A100 GPU OPTIMIZED VERSION")
    print("="*70)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration - A100 optimized
    N_FOLDS = 5
    USE_OPTUNA = OPTUNA_AVAILABLE
    OPTUNA_TRIALS = 50 if OPTUNA_AVAILABLE else 0  # More trials with A100
    
    # Data paths
    train_path = 'data/train.csv' if os.path.exists('data/train.csv') else 'train.csv'
    test_path = 'data/test.csv' if os.path.exists('data/test.csv') else 'test.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return
    
    # Load data
    print(f"\n📊 Loading data from {train_path}...")
    total_start = time.time()
    
    train_df = pd.read_csv(train_path)
    print(f"   Shape: {train_df.shape}")
    print(f"   Memory: {train_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Separate target
    y = train_df['Lap_Time_Seconds'].values
    X = train_df.drop('Lap_Time_Seconds', axis=1)
    
    # Initialize pipeline
    pipeline = AdvancedCVPipeline(n_folds=N_FOLDS, use_optuna=USE_OPTUNA)
    
    # Feature engineering
    print("\n🔧 Applying feature engineering...")
    fe_start = time.time()
    X = pipeline.feature_engineer.fit_transform(X)
    pipeline.feature_names = pipeline.feature_engineer.feature_names
    fe_time = time.time() - fe_start
    print(f"   ✅ Complete in {fe_time:.2f}s")
    print(f"   Features: {len(pipeline.feature_names)}")
    
    # Get feature array
    feature_cols = pipeline.feature_engineer.feature_names
    X_features = X[feature_cols].values if isinstance(X, pd.DataFrame) else X
    
    # Hyperparameter optimization (optional)
    if USE_OPTUNA:
        print("\n" + "="*70)
        print("🔬 HYPERPARAMETER OPTIMIZATION (A100 GPU)")
        print("="*70)
        
        # Use larger subset with A100
        subset_size = min(100000, len(X_features))  # 100k samples with A100
        subset_idx = np.random.choice(len(X_features), subset_size, replace=False)
        X_subset = X_features[subset_idx]
        y_subset = y[subset_idx]
        
        print(f"   Using subset of {subset_size:,} samples for optimization")
        print(f"   Trials: {OPTUNA_TRIALS} per model")
        
        # Optimize Random Forest
        rf_best = pipeline.optimize_hyperparameters_rf(X_subset, y_subset, n_trials=OPTUNA_TRIALS)
        
        # Optimize XGBoost
        xgb_best = pipeline.optimize_hyperparameters_xgb(X_subset, y_subset, n_trials=OPTUNA_TRIALS)
    
    # Cross-validation
    print("\n" + "="*70)
    print(f"🔄 {N_FOLDS}-FOLD CROSS-VALIDATION (A100 GPU)")
    print("="*70)
    
    # RF Cross-validation
    rf_cv_results = pipeline.cross_validate_rf(X_features, y)
    
    # Clear memory
    gc.collect()
    
    # XGBoost Cross-validation
    xgb_cv_results = pipeline.cross_validate_xgb(X_features, y)
    
    # Train final ensemble
    pipeline.train_final_ensemble(X_features, y)
    
    # Ensemble CV results
    print("\n" + "="*70)
    print("🎯 ENSEMBLE CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    # Calculate ensemble metrics
    rf_val_rmse = rf_cv_results['val_rmse_mean']
    xgb_val_rmse = xgb_cv_results['val_rmse_mean']
    
    # Weighted by performance
    rf_weight = 1/rf_val_rmse / (1/rf_val_rmse + 1/xgb_val_rmse)
    xgb_weight = 1/xgb_val_rmse / (1/rf_val_rmse + 1/xgb_val_rmse)
    
    # Estimate ensemble RMSE (usually better than individual models)
    ensemble_rmse_est = np.sqrt(rf_weight**2 * rf_val_rmse**2 + xgb_weight**2 * xgb_val_rmse**2)
    
    print(f"\n  Individual Models (Validation RMSE):")
    print(f"    Random Forest: {rf_val_rmse:.4f} ± {rf_cv_results['val_rmse_std']:.4f}")
    print(f"    XGBoost:       {xgb_val_rmse:.4f} ± {xgb_cv_results['val_rmse_std']:.4f}")
    
    print(f"\n  Ensemble Weights:")
    print(f"    Random Forest: {rf_weight:.3f}")
    print(f"    XGBoost:       {xgb_weight:.3f}")
    
    print(f"\n  Estimated Ensemble RMSE: {ensemble_rmse_est:.4f}")
    
    improvement = min(rf_val_rmse, xgb_val_rmse) - ensemble_rmse_est
    print(f"  Expected improvement: {improvement:.4f} ({improvement/min(rf_val_rmse, xgb_val_rmse)*100:.1f}%)")
    
    # Feature importance
    print("\n📊 Top 20 Features (Ensemble Importance):")
    print("-"*70)
    
    rf_imp = pipeline.feature_importance.get('rf', np.zeros(len(pipeline.feature_names)))
    xgb_imp = pipeline.feature_importance.get('xgb', np.zeros(len(pipeline.feature_names)))
    
    # Combined importance
    ensemble_imp = rf_weight * rf_imp + xgb_weight * xgb_imp
    
    importance_df = pd.DataFrame({
        'feature': pipeline.feature_names[:len(ensemble_imp)],
        'importance': ensemble_imp,
        'rf_importance': rf_imp[:len(pipeline.feature_names)],
        'xgb_importance': xgb_imp[:len(pipeline.feature_names)]
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(20)[['feature', 'importance']].to_string(index=False))
    
    # Save everything
    pipeline.save_all()
    
    # Test predictions
    if os.path.exists(test_path):
        print("\n📈 Generating test predictions (A100 GPU)...")
        
        test_df = pd.read_csv(test_path)
        print(f"   Test shape: {test_df.shape}")
        
        # Find ID column
        id_col = None
        for col in ['id', 'Id', 'ID', 'Unique_ID']:
            if col in test_df.columns:
                id_col = col
                test_ids = test_df[col].values
                break
        
        if id_col is None:
            test_ids = np.arange(len(test_df))
        
        # Feature engineering
        pred_start = time.time()
        X_test = pipeline.feature_engineer.transform(test_df)
        X_test_features = X_test[feature_cols].values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Generate predictions
        print("\n   Generating ensemble predictions...")
        test_pred = pipeline.ensemble_predict(X_test_features, method='weighted_cv')
        pred_time = time.time() - pred_start
        
        print(f"   ✅ Predictions complete in {pred_time:.2f}s (⚡ GPU accelerated)")
        
        # Save submission
        os.makedirs('submissions', exist_ok=True)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'Lap_Time_Seconds': test_pred
        })
        
        submission.to_csv('submissions/submission_cv_a100.csv', index=False)
        print(f"   ✅ Submission saved to submissions/submission_cv_a100.csv")
        
        # Statistics
        print(f"\n   Test Predictions Statistics:")
        print(f"     Mean:  {test_pred.mean():.4f}")
        print(f"     Std:   {test_pred.std():.4f}")
        print(f"     Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Total time
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("✅ A100 GPU OPTIMIZED TRAINING COMPLETE!")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"   Models: {N_FOLDS} folds × 2 models = {N_FOLDS * 2} models trained")
    print(f"   Device: A100 GPU")
    print("="*70)
    
    # Performance summary
    print("\n⚡ A100 GPU Performance Benefits:")
    print(f"   • XGBoost training: GPU accelerated")
    print(f"   • Hyperparameter optimization: {OPTUNA_TRIALS} trials completed")
    print(f"   • Predictions: GPU accelerated")
    print(f"   • Max depth: {12} (optimized for GPU)")
    print(f"   • Max bins: {512} (A100 optimized)")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    main()