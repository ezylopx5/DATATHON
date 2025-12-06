"""
Robust Feature Engineering for Formula Racing
Handles missing columns gracefully and adapts to available data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RacingFeatureEngineer:
    """
    Robust feature engineering that adapts to available columns
    """
    
    def __init__(self):
        self.feature_names = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False
        self.available_columns = None
        
    def create_features(self, df, is_training=True):
        """
        Master feature engineering pipeline with robust error handling
        """
        print("\n🏁 ROBUST FEATURE ENGINEERING")
        print("="*50)
        
        df = df.copy()
        original_shape = df.shape
        
        # Store available columns
        self.available_columns = df.columns.tolist()
        print(f"📊 Input columns: {len(self.available_columns)}")
        
        # Store ID column if exists
        id_col = None
        for col in ['id', 'Id', 'ID', 'Unique_ID']:
            if col in df.columns:
                id_col = col
                id_values = df[col].values
                df = df.drop(columns=[col])
                break
        
        # Create time/sequence features if not present
        df = self._create_time_features(df)
        
        # ========================================
        # PHASE 1: CORE RACING FEATURES
        # ========================================
        print("📊 Phase 1: Core Racing Features")
        df = self._create_speed_features(df)
        df = self._create_circuit_features(df)
        df = self._create_weather_features(df)
        df = self._create_tire_features(df)
        
        # ========================================
        # PHASE 2: DRIVER PERFORMANCE FEATURES
        # ========================================
        print("🏎️ Phase 2: Driver Performance Features")
        df = self._create_driver_features(df)
        df = self._create_driver_form_features(df)
        df = self._create_driver_circuit_history(df)
        
        # ========================================
        # PHASE 3: TEAM/CONSTRUCTOR FEATURES
        # ========================================
        print("🏢 Phase 3: Team/Constructor Features")
        df = self._create_team_features(df)
        df = self._create_team_form_features(df)
        
        # ========================================
        # PHASE 4: DRIVER VS TEAMMATE FEATURES
        # ========================================
        print("⚔️ Phase 4: Driver vs Teammate Features")
        df = self._create_teammate_comparison(df)
        
        # ========================================
        # PHASE 5: QUALIFYING & RACE CRAFT
        # ========================================
        print("🏁 Phase 5: Qualifying & Race Craft Features")
        df = self._create_qualifying_features(df)
        df = self._create_race_craft_features(df)
        
        # ========================================
        # PHASE 6: COMPETITIVE CONTEXT
        # ========================================
        print("🎯 Phase 6: Competitive Context Features")
        df = self._create_competitive_features(df)
        
        # ========================================
        # PHASE 7: ADVANCED INTERACTIONS
        # ========================================
        print("🔬 Phase 7: Interaction Features")
        df = self._create_interaction_features(df)
        
        # ========================================
        # PHASE 8: ROLLING STATISTICS
        # ========================================
        print("📈 Phase 8: Rolling Statistics")
        df = self._create_rolling_features(df)
        
        # ========================================
        # FINAL: ENCODING & CLEANING
        # ========================================
        print("🧹 Final: Encoding & Cleaning")
        df = self._encode_categorical_features(df, is_training)
        df = self._final_cleaning(df)
        
        # Restore ID column
        if id_col:
            df[id_col] = id_values
        
        # Store feature names
        exclude_cols = ['id', 'Id', 'ID', 'Unique_ID', 'Lap_Time_Seconds']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\n✅ Feature Engineering Complete!")
        print(f"   Original features: {original_shape[1]}")
        print(f"   Final features: {len(self.feature_names)}")
        print(f"   New features: {len(self.feature_names) - original_shape[1] + 1}")
        
        return df
    
    def _create_time_features(self, df):
        """Create time-based features if not present"""
        # Look for year column
        year_cols = ['race_year', 'year', 'Year', 'season']
        year_col = None
        for col in year_cols:
            if col in df.columns:
                year_col = col
                if col != 'race_year':
                    df['race_year'] = df[col]
                break
        
        if year_col is None:
            # Try to infer from index or create dummy
            print("  ⚠️ No year column found, creating default")
            df['race_year'] = 2023
        
        # Look for sequence/race number
        seq_cols = ['seq', 'race_num', 'round', 'Round']
        seq_col = None
        for col in seq_cols:
            if col in df.columns:
                seq_col = col
                if col != 'seq':
                    df['seq'] = df[col]
                break
        
        if seq_col is None:
            # Create monotonic sequence
            print("  ⚠️ No sequence column found, creating default")
            df['seq'] = range(1, len(df) + 1)
        
        return df
    
    def _create_speed_features(self, df):
        """Speed-based features"""
        try:
            if 'Formula_Avg_Speed_kmh' in df.columns and 'Len_Circuit_inkm' in df.columns:
                df['Formula_Avg_Speed_kmh'] = df['Formula_Avg_Speed_kmh'].replace(0, 0.001)
                df['Len_Circuit_inkm'] = df['Len_Circuit_inkm'].replace(0, 0.001)
                
                df['theoretical_lap_time'] = (df['Len_Circuit_inkm'] / df['Formula_Avg_Speed_kmh']) * 3600
                df['speed_efficiency'] = df['Formula_Avg_Speed_kmh'] / df['Len_Circuit_inkm']
                df['speed_per_km'] = df['Formula_Avg_Speed_kmh'] / df['Len_Circuit_inkm']
                
                speed_mean = df['Formula_Avg_Speed_kmh'].mean()
                speed_std = df['Formula_Avg_Speed_kmh'].std()
                if speed_std > 0:
                    df['speed_zscore'] = (df['Formula_Avg_Speed_kmh'] - speed_mean) / speed_std
                
                df['speed_percentile'] = df['Formula_Avg_Speed_kmh'].rank(pct=True)
        except Exception as e:
            print(f"  ⚠️ Speed features warning: {e}")
        
        return df
    
    def _create_circuit_features(self, df):
        """Circuit complexity features"""
        try:
            if 'Corners_in_Lap' in df.columns and 'Len_Circuit_inkm' in df.columns:
                df['corner_density'] = df['Corners_in_Lap'] / (df['Len_Circuit_inkm'] + 0.001)
                df['avg_corner_spacing'] = df['Len_Circuit_inkm'] / (df['Corners_in_Lap'] + 1)
                df['circuit_complexity'] = df['Corners_in_Lap'] * np.log1p(df['Len_Circuit_inkm'])
                
                df['is_street_circuit'] = (df['corner_density'] > df['corner_density'].quantile(0.75)).astype(int)
                df['is_high_speed_circuit'] = (df['avg_corner_spacing'] > df['avg_corner_spacing'].quantile(0.75)).astype(int)
                
                if 'Formula_Avg_Speed_kmh' in df.columns:
                    df['corner_speed'] = df['Formula_Avg_Speed_kmh'] / (df['Corners_in_Lap'] + 1)
        except Exception as e:
            print(f"  ⚠️ Circuit features warning: {e}")
        
        return df
    
    def _create_weather_features(self, df):
        """Weather and conditions features"""
        try:
            if 'Track_Temperature_Celsius' in df.columns and 'Ambient_Temperature_Celsius' in df.columns:
                df['temp_diff'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']
                df['temp_ratio'] = df['Track_Temperature_Celsius'] / (df['Ambient_Temperature_Celsius'].abs() + 1)
                df['track_temp_optimal'] = 1 / (1 + np.abs(df['Track_Temperature_Celsius'] - 28))
                df['ambient_temp_optimal'] = 1 / (1 + np.abs(df['Ambient_Temperature_Celsius'] - 22))
            
            if 'Humidity_%' in df.columns:
                df['humidity_factor'] = df['Humidity_%'] / 100
                df['dry_factor'] = 1 - df['humidity_factor']
        except Exception as e:
            print(f"  ⚠️ Weather features warning: {e}")
        
        return df
    
    def _create_tire_features(self, df):
        """Tire strategy features"""
        try:
            if 'Tire_Compound' in df.columns:
                tire_speed_map = {
                    'Soft': 1.05, 'SuperSoft': 1.07, 'UltraSoft': 1.09, 'HyperSoft': 1.10,
                    'Medium': 1.00, 'Hard': 0.95, 'Intermediate': 0.85, 'Wet': 0.75
                }
                df['tire_speed_factor'] = df['Tire_Compound'].map(tire_speed_map).fillna(1.0)
                
                tire_durability_map = {
                    'Soft': 0.7, 'SuperSoft': 0.5, 'UltraSoft': 0.3, 'HyperSoft': 0.2,
                    'Medium': 1.0, 'Hard': 1.5, 'Intermediate': 0.8, 'Wet': 0.6
                }
                df['tire_durability_factor'] = df['Tire_Compound'].map(tire_durability_map).fillna(1.0)
            
            if 'Tire_Degradation_Factor_per_Lap' in df.columns and 'Laps' in df.columns:
                df['total_tire_degradation'] = df['Tire_Degradation_Factor_per_Lap'] * df['Laps']
        except Exception as e:
            print(f"  ⚠️ Tire features warning: {e}")
        
        return df
    
    def _create_driver_features(self, df):
        """Basic driver performance metrics"""
        try:
            if 'wins' in df.columns and 'starts' in df.columns:
                df['starts_safe'] = df['starts'] + 1
                df['win_rate'] = df['wins'] / df['starts_safe']
                
                if 'podiums' in df.columns:
                    df['podium_rate'] = df['podiums'] / df['starts_safe']
                
                if 'with_points' in df.columns:
                    df['points_rate'] = df['with_points'] / df['starts_safe']
                
                if 'finishes' in df.columns:
                    df['dnf_rate'] = 1 - (df['finishes'] / df['starts_safe'])
                
                df['experience_level'] = np.log1p(df['starts'])
                df['experience_squared'] = df['experience_level'] ** 2
                df['rookie_indicator'] = (df['starts'] < 10).astype(int)
                df['veteran_indicator'] = (df['starts'] > 100).astype(int)
                
                if 'Champ_Points' in df.columns:
                    df['points_per_race'] = df['Champ_Points'] / df['starts_safe']
                
                df = df.drop(columns=['starts_safe'])
        except Exception as e:
            print(f"  ⚠️ Driver features warning: {e}")
        
        return df
    
    def _create_driver_form_features(self, df):
        """Driver's recent form"""
        try:
            if 'Rider_ID' in df.columns and 'seq' in df.columns:
                df = df.sort_values(['Rider_ID', 'race_year', 'seq'])
                
                for window in [3, 5]:
                    if 'Formula_Avg_Speed_kmh' in df.columns:
                        df[f'driver_speed_ma{window}'] = df.groupby('Rider_ID')['Formula_Avg_Speed_kmh'].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                    
                    if 'position' in df.columns:
                        df[f'driver_position_ma{window}'] = df.groupby('Rider_ID')['position'].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
        except Exception as e:
            print(f"  ⚠️ Driver form features warning: {e}")
        
        return df
    
    def _create_driver_circuit_history(self, df):
        """Driver's circuit-specific performance"""
        try:
            if 'Rider_ID' in df.columns and 'circuit_name' in df.columns:
                if 'position' in df.columns:
                    df['driver_circuit_avg_position'] = df.groupby(['Rider_ID', 'circuit_name'])['position'].transform('mean')
                    df['driver_circuit_best_position'] = df.groupby(['Rider_ID', 'circuit_name'])['position'].transform('min')
                
                if 'Formula_Avg_Speed_kmh' in df.columns:
                    df['driver_circuit_avg_speed'] = df.groupby(['Rider_ID', 'circuit_name'])['Formula_Avg_Speed_kmh'].transform('mean')
                
                df['driver_circuit_experience'] = df.groupby(['Rider_ID', 'circuit_name']).cumcount()
        except Exception as e:
            print(f"  ⚠️ Driver circuit features warning: {e}")
        
        return df
    
    def _create_team_features(self, df):
        """Team-level features"""
        try:
            team_col = self._get_team_column(df)
            
            if team_col is None and 'Rider_ID' in df.columns:
                df['pseudo_team'] = df.groupby(['race_year', 'seq'])['Rider_ID'].transform(
                    lambda x: x.factorize()[0] // 2
                )
                team_col = 'pseudo_team'
            
            if team_col and 'points' in df.columns:
                df['team_race_points'] = df.groupby(['race_year', 'seq', team_col])['points'].transform('sum')
                df['driver_team_points_share'] = df['points'] / (df['team_race_points'] + 1)
            
            if team_col and 'Formula_Avg_Speed_kmh' in df.columns:
                df['team_avg_speed'] = df.groupby(['race_year', 'seq', team_col])['Formula_Avg_Speed_kmh'].transform('mean')
                df['driver_vs_team_speed'] = df['Formula_Avg_Speed_kmh'] / (df['team_avg_speed'] + 0.001)
            
            if team_col and 'position' in df.columns:
                df['team_best_position'] = df.groupby(['race_year', 'seq', team_col])['position'].transform('min')
                df['is_team_leader'] = (df['position'] == df['team_best_position']).astype(int)
        except Exception as e:
            print(f"  ⚠️ Team features warning: {e}")
        
        return df
    
    def _create_team_form_features(self, df):
        """Team's recent form"""
        try:
            team_col = self._get_team_column(df)
            
            if team_col and 'team_race_points' in df.columns:
                df = df.sort_values(['race_year', 'seq'])
                
                for window in [3, 5]:
                    df[f'team_points_ma{window}'] = df.groupby(team_col)['team_race_points'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
        except Exception as e:
            print(f"  ⚠️ Team form features warning: {e}")
        
        return df
    
    def _create_teammate_comparison(self, df):
        """Comparison with teammates"""
        try:
            team_col = self._get_team_column(df)
            
            if team_col and 'Rider_ID' in df.columns:
                race_groups = ['race_year', 'seq', team_col]
                
                if 'Formula_Avg_Speed_kmh' in df.columns:
                    df['teammate_avg_speed'] = df.groupby(race_groups)['Formula_Avg_Speed_kmh'].transform('mean')
                    df['speed_vs_teammate'] = df['Formula_Avg_Speed_kmh'] / (df['teammate_avg_speed'] + 0.001)
                
                if 'position' in df.columns:
                    df['teammate_avg_position'] = df.groupby(race_groups)['position'].transform('mean')
                    df['position_vs_teammate'] = df['teammate_avg_position'] - df['position']
                    df['beat_teammate'] = (df['position'] < df['teammate_avg_position']).astype(int)
                
                if 'points' in df.columns:
                    df['teammate_total_points'] = df.groupby(race_groups)['points'].transform('sum')
                    df['points_share_in_team'] = df['points'] / (df['teammate_total_points'] + 1)
        except Exception as e:
            print(f"  ⚠️ Teammate features warning: {e}")
        
        return df
    
    def _create_qualifying_features(self, df):
        """Qualifying performance features"""
        try:
            if 'Start_Position' in df.columns:
                df['starting_disadvantage'] = np.log1p(df['Start_Position'])
                df['is_front_row'] = (df['Start_Position'] <= 2).astype(int)
                df['is_top_10'] = (df['Start_Position'] <= 10).astype(int)
                df['qualifying_percentile'] = 1 - (df['Start_Position'] - 1) / df.groupby(['race_year', 'seq'])['Start_Position'].transform('max')
                df['gap_to_pole'] = df['Start_Position'] - 1
                
                if 'Rider_ID' in df.columns:
                    df['driver_avg_qualifying'] = df.groupby('Rider_ID')['Start_Position'].transform('mean')
                    df['qualifying_vs_average'] = df['driver_avg_qualifying'] - df['Start_Position']
        except Exception as e:
            print(f"  ⚠️ Qualifying features warning: {e}")
        
        return df
    
    def _create_race_craft_features(self, df):
        """Race craft and overtaking ability"""
        try:
            if 'Start_Position' in df.columns and 'position' in df.columns:
                df['places_gained'] = df['Start_Position'] - df['position']
                df['improved_position'] = (df['places_gained'] > 0).astype(int)
                df['held_position'] = (df['places_gained'] == 0).astype(int)
                df['lost_position'] = (df['places_gained'] < 0).astype(int)
                df['race_craft_score'] = df['places_gained'] / (df['Start_Position'] + 1)
                df['recovery_drive'] = ((df['Start_Position'] > 10) & (df['position'] <= 10)).astype(int)
                
                if 'Rider_ID' in df.columns:
                    df['driver_avg_places_gained'] = df.groupby('Rider_ID')['places_gained'].transform('mean')
        except Exception as e:
            print(f"  ⚠️ Race craft features warning: {e}")
        
        return df
    
    def _create_competitive_features(self, df):
        """Competitive context"""
        try:
            if 'Formula_Avg_Speed_kmh' in df.columns:
                df['speed_vs_median'] = df['Formula_Avg_Speed_kmh'] / df.groupby(['race_year', 'seq'])['Formula_Avg_Speed_kmh'].transform('median')
                df['speed_vs_leader'] = df['Formula_Avg_Speed_kmh'] / df.groupby(['race_year', 'seq'])['Formula_Avg_Speed_kmh'].transform('max')
                df['field_speed_std'] = df.groupby(['race_year', 'seq'])['Formula_Avg_Speed_kmh'].transform('std')
            
            if 'Champ_Points' in df.columns:
                max_points = df['Champ_Points'].max()
                if max_points > 0:
                    df['points_vs_leader'] = df['Champ_Points'] / max_points
        except Exception as e:
            print(f"  ⚠️ Competitive features warning: {e}")
        
        return df
    
    def _create_interaction_features(self, df):
        """Feature interactions"""
        try:
            if 'driver_vs_team_speed' in df.columns and 'team_avg_speed' in df.columns:
                df['driver_car_synergy'] = df['driver_vs_team_speed'] * df['team_avg_speed'] / 100
            
            if 'experience_level' in df.columns and 'circuit_complexity' in df.columns:
                df['experience_complexity_interaction'] = df['experience_level'] * df['circuit_complexity']
            
            if 'Start_Position' in df.columns and 'driver_avg_places_gained' in df.columns:
                df['overtaking_opportunity'] = df['Start_Position'] * df['driver_avg_places_gained']
        except Exception as e:
            print(f"  ⚠️ Interaction features warning: {e}")
        
        return df
    
    def _create_rolling_features(self, df):
        """Rolling statistics"""
        try:
            if 'Rider_ID' in df.columns:
                df = df.sort_values(['Rider_ID', 'race_year', 'seq'])
                
                for col in ['position', 'points', 'Formula_Avg_Speed_kmh']:
                    if col in df.columns:
                        df[f'{col}_ewma'] = df.groupby('Rider_ID')[col].transform(
                            lambda x: x.ewm(span=5, min_periods=1).mean()
                        )
                
                if 'position' in df.columns:
                    df['position_volatility'] = df.groupby('Rider_ID')['position'].transform(
                        lambda x: x.rolling(5, min_periods=1).std()
                    )
                    df['recent_best_position'] = df.groupby('Rider_ID')['position'].transform(
                        lambda x: x.rolling(5, min_periods=1).min()
                    )
        except Exception as e:
            print(f"  ⚠️ Rolling features warning: {e}")
        
        return df
    
    def _get_team_column(self, df):
        """Find team column"""
        for col in ['Constructor_ID', 'Team_ID', 'team', 'Team', 'constructor', 'pseudo_team']:
            if col in df.columns:
                return col
        return None
    
    def _encode_categorical_features(self, df, is_training):
        """Encode categorical variables"""
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            df[col] = df[col].fillna('missing')
            
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else 'missing'
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def _final_cleaning(self, df):
        """Final data cleaning"""
        df = df.replace([np.inf, -np.inf], 0)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform"""
        self.fitted = True
        return self.create_features(df, is_training=True)
    
    def transform(self, df):
        """Transform only"""
        if not self.fitted:
            print("⚠️ Feature engineer not fitted, using simple transform")
        return self.create_features(df, is_training=False)