import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import mlflow
import mlflow.sklearn
import logging


# Set MLflow tracking URI to the root directory
mlflow.set_tracking_uri("file:///E:/Kiffya_10_acc/Week%208-9/Fraud-Detection/mlruns")

# Set or create a new experiment
mlflow.set_experiment("Fraud_Detection_Experiment")

# Logging setup
logging.basicConfig(level=logging.INFO)


class ModelPipeline:
    """Class to handle data loading, splitting, model training, evaluation, and logging."""

    def __init__(self, dataset_type, path):
        """
        Initialize the pipeline with dataset type and file path.
        
        dataset_type: A string ('creditcard' or 'fraud') to indicate dataset type.
        path: File path for the dataset.
        """
        self.dataset_type = dataset_type
        self.path = path
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load data based on the dataset type."""
        if self.dataset_type == 'creditcard':
            logging.info(f"Loading credit card data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'Class'  # Target column for creditcard dataset

        elif self.dataset_type == 'fraud':
            logging.info(f"Loading fraud data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'class'  # Target column for fraud dataset

        else:
            raise ValueError("Invalid dataset_type! Must be 'creditcard' or 'fraud'")
        
        logging.info("Data loading complete.")

    def split_data(self, test_size=0.3, random_state=42):
        """Split the loaded data into training and test sets."""
        if self.data is not None:
            X = self.data.drop(columns=[self.target])
            y = self.data[self.target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logging.info("Data has been split into train and test sets.")
        else:
            raise ValueError("Data not loaded. Please load the data first.")

    def train_model(self, model, model_name):
        """Train the model with the training data."""
        logging.info(f"Training {model_name} on {self.dataset_type} dataset...")
        model.fit(self.X_train, self.y_train)
        logging.info(f"{model_name} training complete.")

    def evaluate_model(self, model, model_name):
        """Evaluate the model using the test data and return the classification report."""
        logging.info(f"Evaluating {model_name} on {self.dataset_type} dataset...")
        y_pred = model.predict(self.X_test)
        
        report = classification_report(self.y_test, y_pred, output_dict=True)
        logging.info(f"{model_name} evaluation report:\n{classification_report(self.y_test, y_pred)}")
        return report

    def log_model(self, model, model_name, report):
        """Log the model, performance metrics, and save the model artifact to MLflow."""
        logging.info(f"Logging {model_name} to MLflow...")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log model parameters if available
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            # Log classification metrics
            mlflow.log_metrics({
                "precision": report['1']['precision'],
                "recall": report['1']['recall'],
                "f1-score": report['1']['f1-score'],
                "accuracy": report['accuracy']
            })
            
            # Log the model itself
            mlflow.sklearn.log_model(model, f"{self.dataset_type}_{model_name}_model")
            
            logging.info(f"{model_name} has been logged and saved in MLflow.")

    def run_pipeline(self):
        """Run the entire pipeline from loading data to training and logging models."""
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split data
        self.split_data()

        # Step 3: Train and evaluate multiple models
        models = [
            (LogisticRegression(), 'Logistic Regression'),
            (DecisionTreeClassifier(), 'Decision Tree'),
            (RandomForestClassifier(), 'Random Forest'),
            (GradientBoostingClassifier(), 'Gradient Boosting')
        ]
        
        for model, name in models:
            self.train_model(model, name)
            report = self.evaluate_model(model, name)
            self.log_model(model, name, report)