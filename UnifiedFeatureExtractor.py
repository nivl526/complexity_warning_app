class UnifiedFeatureExtractor:
    """Handles extraction of all features from the JSON data."""
    
    def __init__(self, df_item_features):
        self.df_item_features = df_item_features
        self.color_labels = ['blue','brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
        self.shape_labels = ['rectangle', 'round', 'spiky', 'square', 'thin', 'triangle']
    
    def extract_basic_features(self, json_data):
        """Extracts simple numeric features from the JSON."""
        return {
            'duration': json_data.get('duration', 0),
            'assist': json_data.get('assist', 0),
            'has_ease': int(json_data.get('ease', 0) > 0),
            'superEase': json_data.get('superEase', 0),
            'num_type_of_goals': len(json_data.get('goals'))
        }

    def extract_color_and_shape_features(self, goals, board):
        """Extracts color and shape-based features from the items."""
        color_counts_1_goal = {f"number_of_color_1_{color}_goal_items": 0 for color in self.color_labels}
        color_counts_2_goal = {f"number_of_color_2_{color}_goal_items": 0 for color in self.color_labels}
        color_counts_1_board = {f"number_of_color_1_{color}_board_items": 0 for color in self.color_labels}
        color_counts_2_board = {f"number_of_color_2_{color}_board_items": 0 for color in self.color_labels}
        shape_counts = {f"number_of_shape_{shape}_items": 0 for shape in self.shape_labels}

        for item in goals + board:
            item_id = item['id']
            item_count = item['count']
            
            item_row = self.df_item_features[self.df_item_features['item_name'].str.lower() == item_id.lower()]
            if item_row.empty:
                continue

            color_1 = item_row['color_1'].values[0]
            color_2 = item_row['color_2'].values[0]
            shape = item_row['shape'].values[0]

            if item in goals:
                if color_1 in self.color_labels:
                    color_counts_1_goal[f"number_of_color_1_{color_1}_goal_items"] += item_count
                if color_2 in self.color_labels:
                    color_counts_2_goal[f"number_of_color_2_{color_2}_goal_items"] += item_count
            elif item in board:
                if color_1 in self.color_labels:
                    color_counts_1_board[f"number_of_color_1_{color_1}_board_items"] += item_count
                if color_2 in self.color_labels:
                    color_counts_2_board[f"number_of_color_2_{color_2}_board_items"] += item_count

            if shape in self.shape_labels:
                shape_counts[f"number_of_shape_{shape}_items"] += item_count

        return {**color_counts_1_goal, **color_counts_2_goal, **color_counts_1_board, **color_counts_2_board, **shape_counts}

    def calculate_similar_color_features(self, features):
        """Calculates the total number of items with the same color in both the board and goal."""
        
        similar_color_1 = 0
        similar_color_2 = 0

        for color in self.color_labels:
            goal_key_1 = f"number_of_color_1_{color}_goal_items"
            board_key_1 = f"number_of_color_1_{color}_board_items"
            goal_key_2 = f"number_of_color_2_{color}_goal_items"
            board_key_2 = f"number_of_color_2_{color}_board_items"

            # Use max between board and goal for each color
            similar_color_1 += min(max(features.get(goal_key_1, 0), features.get(board_key_1, 0)),
                                  max(features.get(goal_key_1, 0), features.get(board_key_1, 0)))

            similar_color_2 += min(max(features.get(goal_key_2, 0), features.get(board_key_2, 0)),
                                  max(features.get(goal_key_2, 0), features.get(board_key_2, 0)))

        return {
            'num_same_color1_in_board_and_goal': similar_color_1,
            'num_same_color2_in_board_and_goal': similar_color_2
        }
    
    def extract_features_from_json(self, json_data):
        """Extracts all relevant features from the JSON input."""
        basic_features = self.extract_basic_features(json_data)
        goals, board = json_data.get('goals', []), json_data.get('board', [])
        color_shape_features = self.extract_color_and_shape_features(goals, board)

        total_items_goal = sum(item['count'] for item in goals)
        total_items_board = sum(item['count'] for item in board)
        total_items = total_items_goal + total_items_board

        goal_items_per_seconed = total_items_goal / max(1, basic_features['duration'])
        num_goal_items_pct = (total_items_goal / max(1, total_items)) * 100 if total_items > 0 else 0

        # Compute main color proportions (from total amount per color)
        color_1_total_cols = [f"number_of_color_1_{color}_goal_items" for color in self.color_labels] + \
                            [f"number_of_color_1_{color}_board_items" for color in self.color_labels]

        color_totals = {color: 0 for color in self.color_labels}
        
        for color in self.color_labels:
            goal_key = f"number_of_color_1_{color}_goal_items"
            board_key = f"number_of_color_1_{color}_board_items"
            color_totals[color] = color_shape_features.get(goal_key, 0) + color_shape_features.get(board_key, 0)

        max_main_color_proportion = max(color_totals.values(), default=0)

        combined_features = {
            **basic_features,
            **color_shape_features,
            'total_items': total_items,
            'goal_items_per_seconed': goal_items_per_seconed,
            'num_goal_items_pct': num_goal_items_pct,
            'num_goal_items': total_items_goal,
            'max_main_color_proportion': max_main_color_proportion,
            'items_per_seconed':  total_items / basic_features['duration']
        }

        similar_color_features = self.calculate_similar_color_features(combined_features)
        combined_features.update(similar_color_features)
        return combined_features