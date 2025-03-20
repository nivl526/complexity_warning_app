import json 

class JSONParser:
    """Parses raw JSON input and extracts basic features."""
    
    def __init__(self, json):
        self.json = json

    
    def extract_basic_features(self):
        """Extracts simple numeric features from JSON."""
        return {
            'duration': self.json['data'].get('duration', 0),
            'assist': self.json['data'].get('assist', 0),
            'has_ease': int(self.json['data'].get('ease', 0) > 0),
            'superEase': self.json['data'].get('superEase', 0)
        }
    
    def get_items(self):
        """Returns goals and board items from JSON."""
        return self.json['data'].get('goals', []), self.json['data'].get('board', [])