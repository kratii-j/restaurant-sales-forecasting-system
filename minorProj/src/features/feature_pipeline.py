
# Import feature modules
from .temporal_features import TemporalFeatureEngineer
from .behavioral_features import BehavioralFeatureEngineer


def run_feature_pipeline():
    # Step 1: Temporal features
    print("Running temporal feature engineering...")
    temporal_engineer = TemporalFeatureEngineer()
    temporal_engineer.run()

    # Step 2: Behavioral features
    print("Running behavioral feature engineering...")
    behavioral_engineer = BehavioralFeatureEngineer()
    behavioral_engineer.run()

    print("Feature pipeline completed.")

if __name__ == "__main__":
    run_feature_pipeline()
