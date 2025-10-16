import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from joblib import dump, load

class MLPModelTrainer:
    """
    Class for training, optimizing and evaluating an MLP (Multi-Layer Perceptron) model
    for audio click detection.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the MLPModelTrainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_cols = None
        
    def load_data(self, train_path, y_train_path=None, x_submit_path=None):
        """
        Load training and submission data
        
        Args:
            train_path (str): Path to the training data CSV file
            y_train_path (str, optional): Path to the labels CSV file
            x_submit_path (str, optional): Path to the submission data CSV file
            
        Returns:
            tuple: Dataframes of loaded data
        """
        train_df = pd.read_csv(train_path)
        X_submit_df = None
        
        if x_submit_path:
            X_submit_df = pd.read_csv(x_submit_path)
            
        return train_df, X_submit_df
    
    def prepare_data(self, train_df, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            train_df (pd.DataFrame): DataFrame containing training data
            test_size (float): Proportion of data to reserve for validation
            
        Returns:
            tuple: Prepared data for training and validation
        """
        # Identify feature columns
        self.feature_cols = [col for col in train_df.columns 
                             if col not in ['File', 'id', 'pos_label', 'location']]
        
        X = train_df[self.feature_cols].copy()
        y = train_df['pos_label'].copy()
        
        # Display class distribution
        class_counts = np.bincount(y.astype(int))
        print(f"Class distribution: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
        print(f"Class ratio: {class_counts[0]/class_counts[1]:.2f}:1")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, 
            random_state=self.random_state, stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def optimize_model(self, X_train, y_train, param_grid=None, n_splits=5, n_jobs=-1):
        """
        Optimize MLP model hyperparameters using GridSearchCV and StratifiedKFold
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            param_grid (dict, optional): Grid of parameters to test
            n_splits (int): Number of folds for cross-validation
            n_jobs (int): Number of parallel jobs (-1 to use all processors)
            
        Returns:
            MLPClassifier: The best model found
        """
        # Define stratified cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Define hyperparameters to test
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(64, 32), (128, 64), (64, 32, 16)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        
        mlp = MLPClassifier(
            solver='adam',
            batch_size='auto',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state,
            verbose=False
        )
        
        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=n_jobs,
            verbose=2
        )
        
        print("Starting grid search to optimize hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        print("Best hyperparameters found:")
        print(grid_search.best_params_)
        print(f"Best AUC score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_val, y_val):
        """
        Evaluate the model on the validation set
        
        Args:
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            
        Returns:
            tuple: Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        class_report = classification_report(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"ROC-AUC score: {roc_auc:.4f}")
        print("\nConfusion matrix:")
        print(conf_matrix)
        print("\nClassification report:")
        print(class_report)
        
        return y_pred, y_proba
    
    def predict_submission(self, X_submit_df):
        """
        Make predictions on submission data
        
        Args:
            X_submit_df (pd.DataFrame): DataFrame containing submission data
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be trained before prediction")
            
        X_submit = X_submit_df[self.feature_cols].copy()
        X_submit_scaled = self.scaler.transform(X_submit)
        
        submit_proba = self.model.predict_proba(X_submit_scaled)[:, 1]
        submit_pred = self.model.predict(X_submit_scaled)
        
        result_df = pd.DataFrame({
            'File': X_submit_df['File'],
            'Predicted_Label': submit_pred,
            'Probability': submit_proba
        })
        
        return result_df
    
    def create_visualizations(self, X_val, y_val, y_pred, y_proba, result_df, output_dir=None):
        """
        Create visualizations to analyze the model and predictions
        
        Args:
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            y_pred (np.ndarray): Predictions
            y_proba (np.ndarray): Prediction probabilities
            result_df (pd.DataFrame): Prediction results DataFrame
            output_dir (str, optional): Output directory for visualizations
        """
        # Prefix for save paths
        prefix = "" if output_dir is None else f"{output_dir}/"
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f"{prefix}roc_curve.png")
        plt.show()
        
        # Prediction probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(result_df['Probability'], kde=True, bins=20)
        plt.title('Click Detection Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(f"{prefix}probability_distribution.png")
        plt.show()
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(f"{prefix}confusion_matrix.png")
        plt.show()
    
    def save_model(self, model_path='best_model.joblib', scaler_path='StandardScaler.joblib'):
        """
        Save the model and scaler
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be trained before saving")
            
        dump(self.model, model_path)
        dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='best_model.joblib', scaler_path='StandardScaler.joblib'):
        """
        Load the model and scaler
        
        Args:
            model_path (str): Path to the model file
            scaler_path (str): Path to the scaler file
        """
        self.model = load(model_path)
        self.scaler = load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
    
    def run_full_pipeline(self, train_path, x_submit_path, y_submit_path, 
                          save_model=True, create_viz=True, output_dir=None):
        """
        Run the complete training and prediction pipeline
        
        Args:
            train_path (str): Path to the training data CSV file
            x_submit_path (str): Path to the submission data CSV file
            y_submit_path (str): Path to save predictions
            save_model (bool): If True, save the model and scaler
            create_viz (bool): If True, create visualizations
            output_dir (str, optional): Output directory for visualizations
            
        Returns:
            pd.DataFrame: Predictions DataFrame
        """
        # 1. Load data
        print("Loading data...")
        train_df, X_submit_df = self.load_data(train_path, x_submit_path=x_submit_path)
        
        # 2. Prepare data
        print("Preparing data...")
        X_train, X_val, y_train, y_val = self.prepare_data(train_df)
        
        # 3. Optimize model
        print("Optimizing MLP model hyperparameters...")
        self.optimize_model(X_train, y_train)
        
        # 4. Evaluate model
        print("Evaluating optimized model...")
        y_pred, y_proba = self.evaluate_model(X_val, y_val)
        
        # 5. Make predictions on submission data
        print("Making predictions on submission data...")
        result_df = self.predict_submission(X_submit_df)
        
        # 6. Create visualizations
        if create_viz:
            print("Creating visualizations...")
            self.create_visualizations(X_val, y_val, y_pred, y_proba, result_df, output_dir)
        
        # 7. Create submission file
        if 'id' in X_submit_df.columns:
            submission_df = pd.DataFrame({
                'id': X_submit_df['id'],
                'pos_label': result_df['Probability']
            })
        else:
            submission_df = pd.DataFrame({
                'File': result_df['File'],
                'pos_label': result_df['Probability']
            })
        
        submission_df.to_csv(y_submit_path, index=False)
        print(f"Submission file saved to {y_submit_path}")
        
        # Save model and scaler
        if save_model:
            self.save_model()
        
        print("Analysis completed successfully!")
        return submission_df