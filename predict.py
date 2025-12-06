"""
Corrected Prediction Pipeline
Generates predictions in the exact required format
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def make_predictions(test_file='data/test.csv', output_file='submissions/submission.csv'):
    """
    Make predictions using trained models
    Output format: id,Lap_Time_Seconds
    """
    print("🔮 PREDICTION PIPELINE")
    print("="*50)
    
    try:
        # Check if models exist
        model_dir = 'models'
        required_files = {
            'feature_engineer': f'{model_dir}/feature_engineer.pkl',
            'rf': f'{model_dir}/rf_model.pkl',
            'xgb': f'{model_dir}/xgb_model.pkl'
        }
        
        missing_files = [f for f in required_files.values() if not os.path.exists(f)]
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            print("\n💡 Please run training first: python train.py")
            return None
        
        # Load models
        print("\n📂 Loading trained models...")
        feature_engineer = joblib.load(required_files['feature_engineer'])
        print("  ✅ Feature engineer loaded")
        
        rf_model = joblib.load(required_files['rf'])
        print("  ✅ Random Forest loaded")
        
        xgb_model = joblib.load(required_files['xgb'])
        print("  ✅ XGBoost loaded")
        
        # Load test data
        if not os.path.exists(test_file):
            print(f"❌ Test data not found: {test_file}")
            return None
        
        print(f"\n📂 Loading test data: {test_file}")
        df_test = pd.read_csv(test_file)
        print(f"  Test data shape: {df_test.shape}")
        print(f"  Columns: {df_test.columns.tolist()}")
        
        # Find ID column
        id_col = None
        possible_id_cols = ['id', 'Id', 'ID', 'Unique_ID', 'unique_id']
        
        for col in possible_id_cols:
            if col in df_test.columns:
                id_col = col
                break
        
        if id_col is None:
            print("❌ No ID column found in test data")
            print(f"   Available columns: {df_test.columns.tolist()}")
            print("\n💡 Using row indices as IDs")
            test_ids = np.arange(len(df_test))
        else:
            print(f"  ✅ Using ID column: '{id_col}'")
            test_ids = df_test[id_col].values
        
        # Feature engineering
        print("\n🔧 Applying feature engineering...")
        try:
            X_test = feature_engineer.transform(df_test)
            print(f"  ✅ Features transformed: {X_test.shape}")
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Get feature columns (same as training)
        feature_cols = feature_engineer.feature_names
        X_test_features = X_test[feature_cols] if isinstance(X_test, pd.DataFrame) else X_test
        
        # Convert to numpy array
        if isinstance(X_test_features, pd.DataFrame):
            X_test_features = X_test_features.values
        
        print(f"  Final feature matrix: {X_test_features.shape}")
        
        # Make predictions
        print("\n🎯 Generating predictions...")
        
        # Check if RF is GPU model
        try:
            if hasattr(rf_model, '_is_gpu_model') and rf_model._is_gpu_model:
                print("  Detected GPU Random Forest model")
                try:
                    import cudf
                    X_gpu = cudf.DataFrame(X_test_features.astype(np.float32))
                    rf_pred = rf_model.predict(X_gpu).to_numpy()
                except:
                    print("  Warning: GPU prediction failed, using CPU")
                    rf_pred = rf_model.predict(X_test_features)
            else:
                rf_pred = rf_model.predict(X_test_features)
            print("  ✅ Random Forest predictions completed")
        except Exception as e:
            print(f"  ⚠️ Random Forest prediction failed: {e}")
            rf_pred = None
        
        # XGBoost predictions
        try:
            xgb_pred = xgb_model.predict(X_test_features)
            print("  ✅ XGBoost predictions completed")
        except Exception as e:
            print(f"  ⚠️ XGBoost prediction failed: {e}")
            xgb_pred = None
        
        # Ensemble prediction
        if rf_pred is not None and xgb_pred is not None:
            # Use weighted average (favor XGBoost slightly)
            ensemble_pred = 0.3 * rf_pred + 0.7 * xgb_pred
            print("  ✅ Ensemble predictions completed")
        elif xgb_pred is not None:
            print("  ⚠️ Using XGBoost only")
            ensemble_pred = xgb_pred
        elif rf_pred is not None:
            print("  ⚠️ Using Random Forest only")
            ensemble_pred = rf_pred
        else:
            print("❌ Both models failed to predict")
            return None
        
        # Create submission DataFrame in exact format
        print("\n📄 Creating submission file...")
        submission = pd.DataFrame({
            'id': test_ids,
            'Lap_Time_Seconds': ensemble_pred
        })
        
        # Ensure proper data types
        submission['id'] = submission['id'].astype(int)
        submission['Lap_Time_Seconds'] = submission['Lap_Time_Seconds'].astype(float)
        
        # Round to reasonable precision (6 decimal places)
        submission['Lap_Time_Seconds'] = submission['Lap_Time_Seconds'].round(6)
        
        # Create submissions directory
        os.makedirs('submissions', exist_ok=True)
        
        # Save submission
        submission.to_csv(output_file, index=False)
        print(f"  ✅ Submission saved to: {output_file}")
        
        # Show sample of submission
        print("\n📊 Submission Preview (first 10 rows):")
        print(submission.head(10).to_string(index=False))
        
        # Print statistics
        print(f"\n📊 PREDICTION STATISTICS")
        print("-"*50)
        print(f"Number of predictions: {len(submission)}")
        print(f"Mean lap time:         {ensemble_pred.mean():.6f} seconds")
        print(f"Std deviation:         {ensemble_pred.std():.6f} seconds")
        print(f"Min lap time:          {ensemble_pred.min():.6f} seconds")
        print(f"Max lap time:          {ensemble_pred.max():.6f} seconds")
        print(f"Median lap time:       {np.median(ensemble_pred):.6f} seconds")
        
        # Check for anomalies
        print(f"\n🔍 Data Quality Checks:")
        num_zeros = (ensemble_pred == 0).sum()
        num_negative = (ensemble_pred < 0).sum()
        num_outliers = ((ensemble_pred < 60) | (ensemble_pred > 200)).sum()
        
        if num_zeros > 0:
            print(f"  ⚠️ Found {num_zeros} zero predictions")
        if num_negative > 0:
            print(f"  ⚠️ Found {num_negative} negative predictions")
        if num_outliers > 0:
            print(f"  ⚠️ Found {num_outliers} potential outliers (< 60s or > 200s)")
        
        if num_zeros == 0 and num_negative == 0:
            print(f"  ✅ All predictions are positive and non-zero")
        
        # Verify format
        print(f"\n✅ Output Format Verification:")
        print(f"  Columns: {submission.columns.tolist()}")
        print(f"  ID type: {submission['id'].dtype}")
        print(f"  Lap_Time_Seconds type: {submission['Lap_Time_Seconds'].dtype}")
        print(f"  Shape: {submission.shape}")
        
        print("\n" + "="*50)
        print("✅ PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return submission
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_check_submission(filename='submissions/submission.csv'):
    """
    Load and verify submission file format
    """
    print("\n🔍 VERIFYING SUBMISSION FORMAT")
    print("="*50)
    
    try:
        df = pd.read_csv(filename)
        
        print(f"✅ File loaded: {filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Check format
        required_cols = ['id', 'Lap_Time_Seconds']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print("✅ Required columns present")
        
        # Check data types
        print(f"   id dtype: {df['id'].dtype}")
        print(f"   Lap_Time_Seconds dtype: {df['Lap_Time_Seconds'].dtype}")
        
        # Check for missing values
        if df.isnull().any().any():
            print("⚠️ Warning: Found missing values")
            print(df.isnull().sum())
        else:
            print("✅ No missing values")
        
        # Show sample
        print("\n📄 First 10 rows:")
        print(df.head(10).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🏎️  FORMULA RACING LAP TIME PREDICTION")
    print("="*50)
    
    # Make predictions
    submission = make_predictions(
        test_file='data/test.csv',
        output_file='submissions/submission.csv'
    )
    
    if submission is not None:
        # Verify the submission
        load_and_check_submission('submissions/submission.csv')
        
        print("\n🏁 Ready for submission!")
    else:
        print("\n💥 Prediction pipeline failed!")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you've trained models first: python train.py")
        print("   2. Check that test data exists: data/test.csv")
        print("   3. Verify models are saved in: models/")