class Preprocessor:
    """Ensures that the extracted features align with model requirements."""
    
    def __init__(self, model_features):
        self.model_features = model_features

    def transform(self, feature_dict):
        """Aligns feature dictionary with expected model input format."""
        df = pd.DataFrame([feature_dict])
        missing_features = set(self.model_features) - set(df.columns)
        
        # Add missing features with default values (zero)
        for feature in missing_features:
            df[feature] = 0
        
        return df[self.model_features]
