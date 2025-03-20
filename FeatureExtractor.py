class FeatureExtractor:
    """Extracts item-based features using a predefined item features DataFrame."""
    
    def __init__(self, df_item_features, color_labels, shape_labels):
        self.df_item_features = df_item_features
        self.color_labels = color_labels
        self.shape_labels = shape_labels

    def extract_item_features(self, goals, board):
        """Extracts color, shape, and count-based features from items."""
        total_items_goal, total_items_board = 0, 0
        
        # Initialize color and shape feature counts
        color_counts = {f"number_of_color_2_{color}_board_items": 0 for color in self.color_labels}
        shape_counts = {f"number_of_shape_{shape}_items": 0 for shape in self.shape_labels}
        num_type_of_goals = len(goals)
        num_goal_items = sum(item['count'] for item in goals)
        
        # Process both goals and board items
        for item in goals + board:
            item_id = item['id'].lower()
            item_count = item['count']
            
            # Get item features
            item_row = self.df_item_features[self.df_item_features['item_name'].str.lower() == item_id]
            if item_row.empty:
                continue
            
            color_2 = item_row['color_2'].values[0]
            shape = item_row['shape'].values[0]
            
            # Update counts
            if item in board and color_2 in self.color_labels:
                color_counts[f"number_of_color_2_{color_2}_board_items"] += item_count
            if shape in self.shape_labels:
                shape_counts[f"number_of_shape_{shape}_items"] += item_count

            # Total item counts
            if item in goals:
                total_items_goal += item_count
            else:
                total_items_board += item_count
        
        # Compute additional derived features
        features = {
            'total_items': total_items_goal + total_items_board,
            'goal_items_per_seconed': num_goal_items / max(1, self.df_item_features['duration'].mean()),
            'num_goal_items': num_goal_items,
            'num_type_of_goals': num_type_of_goals
        }
        features.update(color_counts)
        features.update(shape_counts)
        
        return features