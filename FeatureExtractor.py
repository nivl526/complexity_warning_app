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
        
        # Count Triplets
        triplets_goals = sum(1 for item in goals if item["count"] == 3)
        triplets_board = sum(1 for item in board if item["count"] == 3)
        
        # Count features directly from item data without recalculating color/shape
        item_counts = {
            'total_items': total_items_goal + total_items_board,
            'goal_items_per_seconed': num_goal_items / max(1, self.df_item_features['duration'].mean()),  # Assuming 'duration' exists in `df_item_features`
            'num_goal_items': num_goal_items,
            'num_type_of_goals': num_type_of_goals,
            'triplets_goals' : triplets_goals,
            'triplets_board' : triplets_board,
            
        }
        
        # Compute 'two_colors_sides' feature
        two_colors_sides = 0
        
        for item in goals + board:
            item_id = item['id']
            item_row = self.df_item_features[self.df_item_features['item_name'].str.lower() == item_id.lower()]
            
            if not item_row.empty:
                # Check if the item has the 'two_colors_sides' feature
                two_colors_sides += item_row['two_colors_sides'].values[0] * item['count']
        
        # Add the 'two_colors_sides' feature to item_counts
        item_counts['two_colors_sides'] = two_colors_sides
        
        # Compute the 'two_colors_sides_pct' feature (percentage)
        if item_counts['total_items'] > 0:
            item_counts['two_colors_sides_pct'] = two_colors_sides / item_counts['total_items']
        else:
            item_counts['two_colors_sides_pct'] = 0
        

        # Return the extracted features directly without unnecessary recalculations
        return item_counts

