import json
import pandas as pd

class FeatureExtractor:
    """Extracts item-based features using a predefined item features DataFrame."""
    
    def __init__(self, df_item_features):
        self.df_item_features = df_item_features

    def extract_item_features(self, goals, board):
        """Extracts count-based features from items (without re-extracting color/shape)."""
        total_items_goal, total_items_board = 0, 0
        
        # Initialize feature counts
        num_type_of_goals = len(goals)
        num_goal_items = sum(item['count'] for item in goals)
        
        # Initialize item counts (these don't depend on color/shape)
        total_items_goal = sum(item['count'] for item in goals)
        total_items_board = sum(item['count'] for item in board)
        
        # Combine goals and board items
        items = goals + board
        
        # Count features directly from item data without recalculating color/shape
        item_counts = {
            'total_items': total_items_goal + total_items_board,
            'goal_items_per_seconed': num_goal_items / max(1, self.df_item_features['duration'].mean()),  # Assuming 'duration' exists in `df_item_features`
            'num_goal_items': num_goal_items,
            'num_type_of_goals': num_type_of_goals
        }
        
        # Return the extracted features directly without unnecessary recalculations
        return item_counts

