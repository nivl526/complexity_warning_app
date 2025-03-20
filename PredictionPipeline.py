from FeatureExtractor import FeatureExtractor
from Preprocessor import Preprocessor
from JSONParser import JSONParser


class PredictionPipeline:
    """End-to-end pipeline for JSON processing and prediction."""
    
    def __init__(self, model, df_item_features, model_features, color_labels, shape_labels):
        self.model = model
        self.df_item_features = df_item_features
        self.feature_extractor = FeatureExtractor(df_item_features, color_labels, shape_labels)
        self.preprocessor = Preprocessor(model_features)

    def predict(self, json_input):
        """Runs the full pipeline: JSON parsing -> feature extraction -> prediction."""
        
        # Step 1: Parse JSON
        parser = JSONParser(json_input)
        basic_features = parser.extract_basic_features()
        goals, board = parser.get_items()
        
        # Step 2: Extract advanced features
        item_features = self.feature_extractor.extract_item_features(goals, board)
        all_features = {**basic_features, **item_features}
        
        # Step 3: Transform for model
        X = self.preprocessor.transform(all_features)
        
        # Step 4: Make prediction
        proba = self.model.predict_proba(X)[:, 1] * 100  # Convert to percentage
        
        # Step 5: Generate alerts
        alert = "No Alert"
        if proba >= 50:
            alert = "RED ALERT"
        elif proba >= 25:
            alert = "EASY ALERT"
        
        return {"probability": proba, "alert": alert}
