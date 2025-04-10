class UnifiedFeatureExtractor:
    """Handles extraction of all features from the JSON data."""
    
    def __init__(self, df_item_features):
        self.df_item_features = df_item_features
        self.color_labels = ['orange', 'white', 'green', 'purple', 'black', 'red', 'yellow', 'brown', 'blue', 'pink', 'grey']
        self.shape_labels = ['round', 'bug', 'other', 'cake', 'thin', 'box', 'flat', 'cylender', 'fat_disk', 'donut']
    
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

    def calculate_color_percentages(self, color_counts, total_items):
        """Calculate the percentage of each color in board or goal items."""
        color_pct = {}
        for key, count in color_counts.items():
            color_pct[key + "_pct"] = (count / total_items) * 100 if total_items > 0 else 0
        return color_pct

    def calculate_shape_percentages(self, shape_counts, total_items):
        """Calculate the percentage of each shape in board or goal items."""
        shape_pct = {}
        for key, count in shape_counts.items():
            # shape_pct[key + "_pct"] = (count / total_items) * 100 if total_items > 0 else 0
            shape_pct[f"number_of_shape_{key}_items_pct"] = (count / total_items) * 100 if total_items > 0 else 0

        return shape_pct

    def calculate_similar_color_features(self, features):
        """Calculates the total number of items with the same color in both the board and goal."""
        similar_color_1 = 0
        similar_color_2 = 0

        for color in self.color_labels:
            goal_key_1 = f"number_of_color_1_{color}_goal_items"
            board_key_1 = f"number_of_color_1_{color}_board_items"
            goal_key_2 = f"number_of_color_2_{color}_goal_items"
            board_key_2 = f"number_of_color_2_{color}_board_items"

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
        
        # Triplets
        triplets_goals = sum(1 for item in goals if item["count"] == 3)
        triplets_board = sum(1 for item in board if item["count"] == 3)
        
        # Compute main color proportions (from total amount per color)
        color_totals = {color: 0 for color in self.color_labels}
        for color in self.color_labels:
            goal_key = f"number_of_color_1_{color}_goal_items"
            board_key = f"number_of_color_1_{color}_board_items"
            color_totals[color] = color_shape_features.get(goal_key, 0) + color_shape_features.get(board_key, 0)

        max_main_color_proportion = max(color_totals.values(), default=0)
        max_pct_main_color_proportion = (max_main_color_proportion / total_items)*100 if total_items > 0 else 0

        # Calculate color_1 percentages
        color_pct_1_goal = self.calculate_color_percentages(
            {k: v for k, v in color_shape_features.items() if "_goal_items" in k and "_color_1_" in k},
            total_items_goal
        )
        color_pct_1_board = self.calculate_color_percentages(
            {k: v for k, v in color_shape_features.items() if "_board_items" in k and "_color_1_" in k},
            total_items  # <-- Use total_items here as requested
        )

        # Calculate shape percentages by combining counts for board and goal items
        combined_shape_counts = {shape: color_shape_features.get(f"number_of_shape_{shape}_items", 0)
                                 for shape in self.shape_labels}

        # Now calculate the percentages
        shape_pct = self.calculate_shape_percentages(combined_shape_counts, total_items)

        combined_features = {
            **basic_features,
            **color_shape_features,
            'total_items': total_items,
            'goal_items_per_seconed': goal_items_per_seconed,
            'num_goal_items_pct': num_goal_items_pct,
            'num_goal_items': total_items_goal,
            'max_main_color_proportion': max_main_color_proportion,
            'max_pct_main_color_proportion': max_pct_main_color_proportion,
            'items_per_seconed': total_items / basic_features['duration'] if basic_features['duration'] > 0 else 0,
            'triplets_board': triplets_board,
            'triplets_goals': triplets_goals,
            **color_pct_1_goal,
            **color_pct_1_board,
            **shape_pct  # Now including the combined shape percentages
        }

        # Compute 'two_colors_sides' feature
        two_colors_sides = 0
        for item in goals + board:
            item_id = item['id']
            item_row = self.df_item_features[self.df_item_features['item_name'].str.lower() == item_id.lower()]
            if not item_row.empty:
                # Check if the item has the 'two_colors_sides' feature
                two_colors_sides += item_row['two_colors_sides'].values[0] * item['count']
        
        combined_features['two_colors_sides'] = two_colors_sides
        
        # Compute the 'two_colors_sides_pct' feature (percentage)
        if combined_features['total_items'] > 0:
            combined_features['two_colors_sides_pct'] = (two_colors_sides / combined_features['total_items'])*100
        else:
            combined_features['two_colors_sides_pct'] = 0

        similar_color_features = self.calculate_similar_color_features(combined_features)
        combined_features.update(similar_color_features)

        combined_features['num_same_color1_in_board_and_goal_pct'] = (combined_features['num_same_color1_in_board_and_goal'] / combined_features['total_items'])*100
        combined_features['num_same_color2_in_board_and_goal_pct'] = (combined_features['num_same_color2_in_board_and_goal'] / combined_features['total_items'])*100

        return combined_features
