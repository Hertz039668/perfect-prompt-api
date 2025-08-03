"""
Prompt Model: Machine learning models for prompt optimization and prediction.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from loguru import logger
import pickle
from pathlib import Path


@dataclass
class ModelPrediction:
    """Prediction result from a prompt model."""
    
    predicted_score: float
    confidence: float
    feature_importance: Dict[str, float]
    model_used: str


@dataclass
class ModelTrainingResult:
    """Result of model training."""
    
    model_name: str
    accuracy_score: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    training_samples: int
    test_score: float


class PromptModel:
    """
    Machine learning model for predicting prompt effectiveness
    and suggesting optimizations.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the PromptModel.
        
        Args:
            model_type: Type of ML model to use ('random_forest', 'gradient_boost', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ML model based on type."""
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boost":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def extract_features(self, prompts: List[str], analyses: List[Any]) -> np.ndarray:
        """
        Extract features from prompts and their analyses.
        
        Args:
            prompts: List of prompt texts
            analyses: List of PromptAnalysis objects
            
        Returns:
            Feature matrix
        """
        features_list = []
        
        for prompt, analysis in zip(prompts, analyses):
            # Basic text features
            text_features = {
                'length': len(prompt),
                'word_count': len(prompt.split()),
                'sentence_count': len([s for s in prompt.split('.') if s.strip()]),
                'avg_word_length': np.mean([len(word) for word in prompt.split()]),
                'question_count': prompt.count('?'),
                'exclamation_count': prompt.count('!'),
                'uppercase_ratio': sum(1 for c in prompt if c.isupper()) / len(prompt) if prompt else 0,
            }
            
            # Analysis-based features
            analysis_features = {
                'complexity_score': analysis.metrics.complexity_score,
                'clarity_score': analysis.metrics.clarity_score,
                'specificity_score': analysis.metrics.specificity_score,
                'sentiment_score': analysis.metrics.sentiment_score,
                'readability_score': analysis.metrics.readability_score,
                'semantic_density': analysis.metrics.semantic_density,
                'instruction_clarity': analysis.metrics.instruction_clarity,
                'context_richness': analysis.metrics.context_richness,
            }
            
            # Pattern-based features
            pattern_features = {
                'has_role_play': any('role' in p.lower() for p in analysis.identified_patterns),
                'has_structure': any('structure' in p.lower() for p in analysis.identified_patterns),
                'has_examples': any('example' in p.lower() for p in analysis.identified_patterns),
                'has_format': any('format' in p.lower() for p in analysis.identified_patterns),
            }
            
            # Combine all features
            combined_features = {**text_features, **analysis_features, **pattern_features}
            features_list.append(combined_features)
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(features_list)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Handle text vectorization separately if needed
        if hasattr(self, 'include_text_features') and self.include_text_features:
            text_vectors = self.text_vectorizer.fit_transform(prompts)
            text_feature_names = [f"text_feature_{i}" for i in range(text_vectors.shape[1])]
            self.feature_names.extend(text_feature_names)
            
            # Combine numerical and text features
            numerical_features = df.values
            combined_features = np.hstack([numerical_features, text_vectors.toarray()])
            return combined_features
        
        return df.values
    
    def train(
        self, 
        prompts: List[str], 
        analyses: List[Any], 
        target_scores: List[float],
        test_size: float = 0.2
    ) -> ModelTrainingResult:
        """
        Train the model on prompt data.
        
        Args:
            prompts: List of prompt texts
            analyses: List of PromptAnalysis objects
            target_scores: Target effectiveness scores
            test_size: Fraction of data to use for testing
            
        Returns:
            Training result with metrics
        """
        logger.info(f"Training {self.model_type} model on {len(prompts)} samples")
        
        # Extract features
        X = self.extract_features(prompts, analyses)
        y = np.array(target_scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        test_score = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='r2'
        )
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                feature_importance[name] = float(importance)
        
        self.is_trained = True
        
        result = ModelTrainingResult(
            model_name=self.model_type,
            accuracy_score=test_score,
            cross_val_scores=cv_scores.tolist(),
            feature_importance=feature_importance,
            training_samples=len(X_train),
            test_score=test_score
        )
        
        logger.info(f"Model trained successfully. Test R² score: {test_score:.4f}")
        return result
    
    def predict(self, prompt: str, analysis: Any) -> ModelPrediction:
        """
        Predict effectiveness score for a prompt.
        
        Args:
            prompt: The prompt text
            analysis: PromptAnalysis object
            
        Returns:
            Model prediction with confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        X = self.extract_features([prompt], [analysis])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predicted_score = self.model.predict(X_scaled)[0]
        
        # Calculate confidence (simplified)
        confidence = min(1.0, max(0.0, 1.0 - abs(predicted_score - 0.5) * 2))
        
        # Feature importance for this prediction
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                feature_importance[name] = float(importance)
        
        return ModelPrediction(
            predicted_score=float(predicted_score),
            confidence=confidence,
            feature_importance=feature_importance,
            model_used=self.model_type
        )
    
    def predict_batch(self, prompts: List[str], analyses: List[Any]) -> List[ModelPrediction]:
        """
        Predict effectiveness scores for multiple prompts.
        
        Args:
            prompts: List of prompt texts
            analyses: List of PromptAnalysis objects
            
        Returns:
            List of model predictions
        """
        predictions = []
        for prompt, analysis in zip(prompts, analyses):
            prediction = self.predict(prompt, analysis)
            predictions.append(prediction)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model type does not support feature importance")
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'text_vectorizer': self.text_vectorizer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.text_vectorizer = model_data['text_vectorizer']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate_model(
        self, 
        prompts: List[str], 
        analyses: List[Any], 
        true_scores: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            prompts: List of prompt texts
            analyses: List of PromptAnalysis objects
            true_scores: True effectiveness scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict_batch(prompts, analyses)
        predicted_scores = [p.predicted_score for p in predictions]
        
        # Calculate metrics
        mse = mean_squared_error(true_scores, predicted_scores)
        r2 = r2_score(true_scores, predicted_scores)
        mae = np.mean(np.abs(np.array(true_scores) - np.array(predicted_scores)))
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2_score': r2,
            'accuracy': r2  # For compatibility
        }
    
    def suggest_improvements(
        self, 
        prompt: str, 
        analysis: Any, 
        target_score: float = 0.8
    ) -> List[str]:
        """
        Suggest improvements to reach target effectiveness score.
        
        Args:
            prompt: The prompt to improve
            analysis: Current PromptAnalysis
            target_score: Target effectiveness score
            
        Returns:
            List of improvement suggestions
        """
        current_prediction = self.predict(prompt, analysis)
        current_score = current_prediction.predicted_score
        
        if current_score >= target_score:
            return ["Prompt already meets target effectiveness score"]
        
        suggestions = []
        feature_importance = current_prediction.feature_importance
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Generate suggestions based on most important features
        for feature_name, importance in sorted_features[:5]:
            if 'clarity' in feature_name and analysis.metrics.clarity_score < 0.7:
                suggestions.append(
                    "Improve clarity by using more specific action verbs and reducing ambiguous language"
                )
            elif 'specificity' in feature_name and analysis.metrics.specificity_score < 0.6:
                suggestions.append(
                    "Add more specific details, examples, or constraints to improve precision"
                )
            elif 'length' in feature_name and len(prompt) > 300:
                suggestions.append(
                    "Consider shortening the prompt while maintaining key information"
                )
            elif 'complexity' in feature_name and analysis.metrics.complexity_score > 0.8:
                suggestions.append(
                    "Simplify sentence structure to reduce complexity"
                )
            elif 'instruction_clarity' in feature_name and analysis.metrics.instruction_clarity < 0.6:
                suggestions.append(
                    "Make instructions more explicit with clear action words"
                )
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        return unique_suggestions[:5]  # Return top 5 suggestions


class EnsemblePromptModel:
    """
    Ensemble model combining multiple PromptModel instances
    for improved prediction accuracy.
    """
    
    def __init__(self, model_types: List[str] = None):
        """
        Initialize ensemble model.
        
        Args:
            model_types: List of model types to include in ensemble
        """
        if model_types is None:
            model_types = ["random_forest", "gradient_boost", "linear"]
        
        self.models = {
            model_type: PromptModel(model_type) 
            for model_type in model_types
        }
        self.weights = {model_type: 1.0 for model_type in model_types}
        self.is_trained = False
    
    def train(
        self, 
        prompts: List[str], 
        analyses: List[Any], 
        target_scores: List[float]
    ) -> Dict[str, ModelTrainingResult]:
        """Train all models in the ensemble."""
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} in ensemble")
            result = model.train(prompts, analyses, target_scores)
            results[model_name] = result
            
            # Update weight based on performance
            self.weights[model_name] = max(0.1, result.accuracy_score)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for model_name in self.weights:
            self.weights[model_name] /= total_weight
        
        self.is_trained = True
        return results
    
    def predict(self, prompt: str, analysis: Any) -> ModelPrediction:
        """Make ensemble prediction."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weighted_score = 0.0
        combined_confidence = 0.0
        
        for model_name, model in self.models.items():
            pred = model.predict(prompt, analysis)
            predictions.append(pred)
            
            weight = self.weights[model_name]
            weighted_score += pred.predicted_score * weight
            combined_confidence += pred.confidence * weight
        
        # Combine feature importance
        combined_importance = {}
        for pred in predictions:
            for feature, importance in pred.feature_importance.items():
                if feature not in combined_importance:
                    combined_importance[feature] = 0.0
                combined_importance[feature] += importance / len(predictions)
        
        return ModelPrediction(
            predicted_score=weighted_score,
            confidence=combined_confidence,
            feature_importance=combined_importance,
            model_used="ensemble"
        )
    
    def save_ensemble(self, directory: str) -> None:
        """Save ensemble models to directory."""
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = dir_path / f"{model_name}_model.joblib"
            model.save_model(str(filepath))
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        
        metadata_path = dir_path / "ensemble_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load_ensemble(self, directory: str) -> None:
        """Load ensemble models from directory."""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {directory}")
        
        # Load individual models
        for model_name, model in self.models.items():
            filepath = dir_path / f"{model_name}_model.joblib"
            if filepath.exists():
                model.load_model(str(filepath))
        
        # Load ensemble metadata
        metadata_path = dir_path / "ensemble_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.weights = metadata['weights']
            self.is_trained = metadata['is_trained']
        
        logger.info(f"Ensemble loaded from {directory}")
