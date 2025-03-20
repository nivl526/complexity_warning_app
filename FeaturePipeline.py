from FeatureExtractor import FeatureExtractor
from JSONParser import JSONParser

class FeaturePipeline:
    """Combines basic features and item-based features into a final feature set."""
    
    def __init__(self, df_item_features, color_labels, shape_labels):
        self.feature_extractor = FeatureExtractor(df_item_features, color_labels, shape_labels)
    
    def extract_features_from_json(self, json_data):
        """Extracts all relevant features from the JSON input."""
        # Step 1: Parse the JSON data
        json_parser = JSONParser(json_data)
        
        # Step 2: Extract basic features
        basic_features = json_parser.extract_basic_features()
        
        # Step 3: Extract item-based features from goals and board
        goals, board = json_parser.get_items()
        item_features = self.feature_extractor.extract_item_features(goals, board)
        
        # Step 4: Combine both sets of features
        combined_features = {**basic_features, **item_features}
        
        return combined_features
