import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
import logging


@dataclass
class ModelConfig:
    name: str
    model: Any
    params: Dict[str, Any]


@dataclass
class EvaluationMetrics:
    accuracy: float
    tpr: float
    fpr: float
    confusion_matrix: np.ndarray
    training_time: float


class SeizureClassifier:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.models = self._get_model_configs()
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    def _get_model_configs(self) -> List[ModelConfig]:
        return [
            ModelConfig(
                "MLP",
                MLPClassifier,
                {
                    "hidden_layer_sizes": (10, 10),
                    "max_iter": 500,
                    "activation": "relu",
                    "solver": "adam",
                },
            ),
            ModelConfig(
                "SVM",
                SVC,
                {"kernel": "rbf", "class_weight": {1: 100}, "random_state": 0},
            ),
            ModelConfig(
                "RandomForest",
                RandomForestClassifier,
                {"min_samples_split": 10, "class_weight": {1: 100}, "random_state": 0},
            ),
            ModelConfig("AdaBoost", AdaBoostClassifier, {"random_state": 0}),
            ModelConfig("KNN", KNeighborsClassifier, {"n_neighbors": 2}),
        ]

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.info("Loading data")
        data = pd.read_csv(self.input_file)

        X = data.drop(["seizure", "start_time", "subject"], axis=1, errors="ignore")
        y = data["seizure"].values

        selector = VarianceThreshold()
        X = selector.fit_transform(X)

        X = normalize(X)

        self.logger.info(f"Dataset shape: {X.shape}")
        self.logger.info(f"Seizure samples: {np.sum(y)}")
        return X, y

    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
        kf = KFold(n_splits=5)
        cv_metrics = []
        self.logger.info("Starting cross-validation for model")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            self.logger.info(f"Split: {train_idx} {val_idx}")
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            self.logger.info(f"Fold {fold} train shape: {X_fold_train.shape}")
            self.logger.info(f"Fold {fold} validation shape: {X_fold_val.shape}")

            model.fit(X_fold_train, y_fold_train)

            y_pred = model.predict(X_fold_val)

            tn, fp, fn, tp = confusion_matrix(y_fold_val, y_pred).ravel()

            fold_metrics = EvaluationMetrics(
                accuracy=(tp + tn) / (tp + tn + fp + fn),
                tpr=tp / (tp + fn) if (tp + fn) > 0 else 0,
                fpr=fp / (fp + tn) if (fp + tn) > 0 else 0,
                confusion_matrix=confusion_matrix(y_fold_val, y_pred),
                training_time=0,
            )

            cv_metrics.append(fold_metrics)
            self.logger.info(
                f"Fold {fold} Metrics - Accuracy: {fold_metrics.accuracy}, TPR: {fold_metrics.tpr}"
            )

        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        training_time = time.time() - start_time

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        test_metrics = EvaluationMetrics(
            accuracy=(tp + tn) / (tp + tn + fp + fn),
            tpr=tp / (tp + fn) if (tp + fn) > 0 else 0,
            fpr=fp / (fp + tn) if (fp + tn) > 0 else 0,
            confusion_matrix=confusion_matrix(y_test, y_pred),
            training_time=training_time,
        )

        self.logger.info("Test Set Metrics:")
        self.logger.info(f"Accuracy: {test_metrics.accuracy}")
        self.logger.info(f"TPR: {test_metrics.tpr}")
        self.logger.info(f"Training Time: {test_metrics.training_time} seconds")

        return cv_metrics, test_metrics

    def save_results(
        self,
        model_name: str,
        cv_metrics: List[EvaluationMetrics],
        test_metrics: EvaluationMetrics,
    ):
        results = {
            "model": model_name,
            "cv_accuracy": np.mean([m.accuracy for m in cv_metrics]),
            "cv_tpr": np.mean([m.tpr for m in cv_metrics]),
            "cv_fpr": np.mean([m.fpr for m in cv_metrics]),
            "test_accuracy": test_metrics.accuracy,
            "test_tpr": test_metrics.tpr,
            "test_fpr": test_metrics.fpr,
            "training_time": test_metrics.training_time,
        }

        df = pd.DataFrame([results])
        output_file = os.path.join(self.output_dir, f"{model_name}_results.csv")
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")

    def run_classification(self):
        X, y = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        all_results = []
        for config in self.models:
            self.logger.info(f"\nTraining {config.name}")
            try:
                model = config.model(**config.params)
                cv_metrics, test_metrics = self.evaluate_model(
                    model, X_train, X_test, y_train, y_test
                )
                self.save_results(config.name, cv_metrics, test_metrics)
                all_results.append({"model": config.name, "metrics": test_metrics})
            except Exception as e:
                self.logger.error(f"Error training {config.name}: {str(e)}")

        combined_results = pd.concat(
            [
                pd.read_csv(
                    os.path.join(self.output_dir, f"{model['model']}_results.csv")
                )
                for model in all_results
            ]
        )
        combined_results.to_csv(
            os.path.join(self.output_dir, "all_results.csv"), index=False
        )
        self.logger.info("\nAll results saved successfully!")


def main():
    classifier = SeizureClassifier(
        input_file="dataset/subjects.csv", output_dir="output/classification_results"
    )
    classifier.run_classification()


if __name__ == "__main__":
    main()
