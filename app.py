import streamlit as st
import json
import joblib
import pandas as pd
from UnifiedFeatureExtractor import UnifiedFeatureExtractor
from xgboost import XGBClassifier

PASSWORD = st.secrets["secrets"]["password"]


def load_models(item_pack):
    """Load the pretrained XGBoost classifier and corresponding items DataFrame."""
    st.write("Loading models...")

    # Load classifier
    classifier = joblib.load("models/xgb_classifier.pkl")

    # Select item pack
    file_map = {
        "new": "items_data/new_items_colors_and_shapes.csv",
        "old": "items_data/old_items_colors_and_shapes.csv"
    }
    items_file = file_map.get(item_pack, "items_data/new_items_colors_and_shapes.csv")
    items_df = pd.read_csv(items_file)

    st.write(f"Models loaded successfully! Using '{item_pack}' item pack.")
    return classifier, items_df



def main():
  
    st.title("Level Complexity Prediction")
 
    # Dropdown for selecting item pack
    item_pack = st.selectbox("Select Item Pack:", ["new", "old"])

    classifier, items_df = load_models(item_pack)
    feature_extractor = UnifiedFeatureExtractor(items_df)

    input_json = st.text_area("Enter Level JSON:", height=300)

    if st.button("Predict Complexity"):
        if input_json:
            try:
                level_data = json.loads(input_json)  # Convert JSON string to dict
                st.write("✅ JSON data loaded successfully!")

                # Extract features using UnifiedFeatureExtractor
                st.write("🔄 Extracting features...")
                extracted_features = feature_extractor.extract_features_from_json(level_data)

                # Debug: Show the full extracted features dict
                st.subheader("🛠 Full Extracted Features Dict (Debugging)")
                st.json(extracted_features)

                # Select only the required model features
                model_features = [
                    'number_of_color_1_red_board_items_pct','number_of_color_1_yellow_board_items_pct', 'number_of_color_1_brown_board_items_pct',
                    'number_of_color_1_brown_goal_items_pct','items_per_seconed','max_pct_main_color_proportion',
                    'num_type_of_goals','num_goal_items_pct','num_same_color1_in_board_and_goal_pct','number_of_shape_box_items_pct',
                    'number_of_shape_round_items_pct','triplets_goals','triplets_board',
                    'two_colors_sides_pct'
                ]
                features_dict = {key: extracted_features[key] for key in model_features}
                features_df = pd.DataFrame([features_dict])

                # Display extracted features
                st.subheader("📊 Extracted Features for Prediction:")
                st.dataframe(features_df.T, use_container_width=True)

                # Make prediction
                st.write("🚀 Making prediction...")
                prediction = classifier.predict(features_df)[0]
                proba_1 = classifier.predict_proba(features_df)[:, 1][0] * 100  # Probability for class 1

                # Add probability-based messages
                if proba_1 >= 50:
                    st.warning("⚠ High chance for 8+ complexity")
                elif 30 <= proba_1 < 50:
                    st.info("🔵 Medium chance for 8+ complexity")
                else:
                    st.info("✅ Small chance for 8+ complexity")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.error("🚨 Please provide a valid JSON input.")

if __name__ == "__main__":
    main()