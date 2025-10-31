# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

class CropRecommender:
    """
    A class to handle crop recommendation, including data loading,
    model training, analysis, and prediction.
    """
    def __init__(self, dataset_path='Crop_recommendation1.csv'):
        """
        Constructor for the class.
        
        Args:
            dataset_path (str): The file path for the crop recommendation dataset.
        """
        self.dataset_path = dataset_path
        self.data = None
        self.model = None
        
        self._load_data()
        self._train_model()

    def _load_data(self):
        """Loads the dataset from an external CSV, with a remote URL fallback."""
        try:
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully from '{self.dataset_path}'.")
        except FileNotFoundError:
            print(f"Error: The file at '{self.dataset_path}' was not found.")
            print("Attempting to load from a remote URL as a fallback...")
            url = 'https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/crop_recommendation.csv'
            try:
                self.data = pd.read_csv(url)
                print("Dataset loaded successfully from remote URL.")
            except Exception as e:
                print(f"Failed to load data from remote URL as well. Error: {e}")
                self.data = None
        
        if self.data is not None:
            print("\n--- First 5 rows of the dataset: ---")
            print(self.data.head())

    def _train_model(self):
        """Prepares data and trains a RandomForestClassifier model."""
        if self.data is None:
            print("Data not loaded. Cannot train the model.")
            return

        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        target = 'label'
        
        X = self.data[features]
        y = self.data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel trained successfully. Accuracy on test data: {accuracy * 100:.2f}%")

    # --- Modular Visualization Functions ---

    def show_descriptive_statistics(self):
        """Calculates and prints a statistical summary of the dataset."""
        if self.data is None: return
        print("\n--- 1. Descriptive Statistics ---")
        print(self.data.describe())

    def visualize_feature_distributions(self):
        """Generates a separate histogram for each environmental factor."""
        if self.data is None: return
        print("\n--- 2. Feature Distribution Histograms ---")
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in features:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.data[feature], bins=20, kde=True)
            plt.title(f'Distribution of {feature.capitalize()}', size=16)
            plt.xlabel(feature.capitalize())
            plt.ylabel('Frequency')
            plt.show()
            input(f"Finished showing plot for '{feature.capitalize()}'. Press Enter to continue...")
        print("All distribution plots have been shown.")


    def visualize_correlation_heatmap(self):
        """Generates a heatmap to show the correlation between features."""
        if self.data is None: return
        print("\n--- 3. Correlation Heatmap ---")
        plt.figure(figsize=(10, 8))
        correlation = self.data.drop('label', axis=1).corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Features', size=16)
        plt.show()

    def visualize_conditions_by_crop(self):
        """Generates box plots to compare the range of conditions for each crop."""
        if self.data is None: return
        print("\n--- 4. Box Plots for Crop Conditions ---")
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in features:
            plt.figure(figsize=(15, 8))
            sns.boxplot(x='label', y=feature, data=self.data)
            plt.title(f'Distribution of {feature.capitalize()} for Each Crop', size=16)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            input(f"Finished showing plots for '{feature.capitalize()}'. Press Enter to continue...")
        print("All box plots have been shown.")

    # --- Recommendation Functions ---

    def recommend_crop(self, input_conditions):
        """Recommends a crop based on a dictionary of input conditions."""
        if self.model is None: return "Model is not trained yet."
        input_df = pd.DataFrame([input_conditions])
        ordered_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = input_df[ordered_features]
        prediction = self.model.predict(input_df)
        return prediction[0]

    def get_conditions_for_crop(self, crop_name):
        """Retrieves the average environmental conditions for a specific crop."""
        if self.data is None: return "Data not loaded."
        crop_name_lower = crop_name.lower()
        if crop_name_lower not in self.data['label'].str.lower().unique():
            return f"Error: Crop '{crop_name}' not found in the dataset."
        crop_conditions = self.data[self.data['label'].str.lower() == crop_name_lower].mean(numeric_only=True)
        return crop_conditions.to_dict()

# --- Main Execution Block ---
if __name__ == '__main__':
    recommender = CropRecommender()

    if recommender.data is not None:
        # --- Guided Exploratory Data Analysis (EDA) ---
        print("\nStarting Guided Exploratory Data Analysis.")
        
        recommender.show_descriptive_statistics()
        input("\nPress Enter to see the Feature Distribution Histograms...")

        recommender.visualize_feature_distributions()
        input("\nPress Enter to see the Correlation Heatmap...")
        
        recommender.visualize_correlation_heatmap()
        input("\nPress Enter to see the Box Plots for each feature by crop...")

        recommender.visualize_conditions_by_crop()
        
        print("\n--- Guided EDA Complete ---")

        # --- Example Predictions ---
        print("\n--- Example 1: Get a Crop Recommendation ---")
        sample_conditions = {
            'N': 80, 'P': 40, 'K': 40,
            'temperature': 25.5, 'humidity': 80.0,
            'ph': 6.8, 'rainfall': 175.0
        }
        recommended_crop = recommender.recommend_crop(sample_conditions)
        print(f"For the input conditions: {sample_conditions}")
        print(f"==> The Recommended Crop is: {recommended_crop.capitalize()}")

        print("\n--- Example 2: Get Average Conditions for a Specific Crop ---")
        crop_to_check = 'Maize'
        conditions = recommender.get_conditions_for_crop(crop_to_check)
        print(f"The average ideal conditions for '{crop_to_check}' are:")
        if isinstance(conditions, dict):
            for key, value in conditions.items():
                print(f"  - {key.capitalize()}: {value:.2f}")
        else:
            print(conditions)

