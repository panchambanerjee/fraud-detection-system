"""
Model comparison framework for fraud detection system.

This module provides comprehensive comparison between different ML models,
performance analysis, and ensemble methods for optimal fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import logging
from dataclasses import dataclass
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelComparisonResult:
    """Container for model comparison results."""
    model_name: str
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    true_positive_rate: float
    inference_time_ms: float
    model_size_mb: float
    training_time_seconds: float

class ModelComparison:
    """
    Comprehensive model comparison framework for fraud detection.
    
    This class compares the performance of different ML models:
    - Logistic Regression (baseline)
    - XGBoost (tree-based)
    - Deep Neural Network (neural network)
    
    Provides detailed analysis and visualization of results.
    """
    
    def __init__(self):
        """Initialize the model comparison framework."""
        self.models = {}
        self.comparison_results = {}
        self.ensemble_predictions = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Model comparison framework initialized")
    
    def add_model(self, model_name: str, model_instance: Any):
        """
        Add a trained model to the comparison framework.
        
        Args:
            model_name: Name of the model
            model_instance: Trained model instance
        """
        self.models[model_name] = model_instance
        logger.info("Added model: %s", model_name)
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelComparisonResult]:
        """
        Compare all models on the same test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of comparison results for each model
        """
        logger.info("Starting model comparison on test set")
        
        comparison_results = {}
        
        for model_name, model in self.models.items():
            logger.info("Evaluating model: %s", model_name)
            
            try:
                # Evaluate model
                performance = model.evaluate(X_test, y_test)
                
                # Get model info
                model_info = model.get_model_info()
                
                # Create comparison result
                result = ModelComparisonResult(
                    model_name=model_name,
                    roc_auc=performance.roc_auc,
                    precision=performance.precision,
                    recall=performance.recall,
                    f1_score=performance.f1_score,
                    false_positive_rate=performance.false_positive_rate,
                    true_positive_rate=performance.true_positive_rate,
                    inference_time_ms=performance.inference_time_ms,
                    model_size_mb=model_info.get('model_size_mb', 0.0),
                    training_time_seconds=model_info.get('training_time_seconds', 0.0)
                )
                
                comparison_results[model_name] = result
                
                logger.info("Model %s - ROC-AUC: %.4f, Inference: %.2fms", 
                           model_name, performance.roc_auc, performance.inference_time_ms)
                
            except Exception as e:
                logger.error("Error evaluating model %s: %s", model_name, str(e))
                continue
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def create_performance_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of all model performances.
        
        Returns:
            DataFrame with performance metrics for all models
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run compare_models first.")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        summary_data = []
        for model_name, result in self.comparison_results.items():
            summary_data.append({
                'Model': model_name,
                'ROC-AUC': result.roc_auc,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1_score,
                'FPR at 1%': result.false_positive_rate,
                'TPR at 1% FPR': result.true_positive_rate,
                'Inference Time (ms)': result.inference_time_ms,
                'Model Size (MB)': result.model_size_mb,
                'Training Time (s)': result.training_time_seconds
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by ROC-AUC (descending)
        summary_df = summary_df.sort_values('ROC-AUC', ascending=False)
        
        return summary_df
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series, 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get predictions
                probabilities = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, probabilities)
                auc = self.comparison_results[model_name].roc_auc
                
                # Plot ROC curve
                color = colors[i % len(colors)]
                ax.plot(fpr, tpr, color=color, linewidth=2, 
                       label=f'{model_name} (AUC = {auc:.3f})')
                
            except Exception as e:
                logger.error("Error plotting ROC curve for %s: %s", model_name, str(e))
                continue
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Customize plot
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("ROC curves plot saved to %s", save_path)
        
        return fig
    
    def plot_precision_recall_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curves for all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                # Get predictions
                probabilities = model.predict_proba(X_test)[:, 1]
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, probabilities)
                
                # Plot precision-recall curve
                color = colors[i % len(colors)]
                ax.plot(recall, precision, color=color, linewidth=2, 
                       label=f'{model_name}')
                
            except Exception as e:
                logger.error("Error plotting PR curve for %s: %s", model_name, str(e))
                continue
        
        # Customize plot
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Precision-recall curves plot saved to %s", save_path)
        
        return fig
    
    def plot_performance_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot performance metrics comparison.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run compare_models first.")
            return plt.figure()
        
        # Prepare data for plotting
        metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
        model_names = list(self.comparison_results.keys())
        
        # Create data matrix
        data_matrix = []
        for metric in metrics:
            row = []
            for model_name in model_names:
                result = self.comparison_results[model_name]
                if metric == 'ROC-AUC':
                    row.append(result.roc_auc)
                elif metric == 'Precision':
                    row.append(result.precision)
                elif metric == 'Recall':
                    row.append(result.recall)
                elif metric == 'F1-Score':
                    row.append(result.f1_score)
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(data_matrix, 
                   xticklabels=model_names,
                   yticklabels=metrics,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=0.5,
                   ax=ax)
        
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Metrics', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Performance metrics plot saved to %s", save_path)
        
        return fig
    
    def plot_inference_time_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot inference time comparison.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run compare_models first.")
            return plt.figure()
        
        # Prepare data
        model_names = list(self.comparison_results.keys())
        inference_times = [self.comparison_results[name].inference_time_ms for name in model_names]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(model_names, inference_times, color=['skyblue', 'lightgreen', 'lightcoral'])
        
        # Add value labels on bars
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_ylabel('Inference Time (ms)', fontsize=12)
        ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add 100ms threshold line
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100ms Target')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Inference time comparison plot saved to %s", save_path)
        
        return fig
    
    def create_ensemble_predictions(self, X_test: pd.DataFrame, 
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Create ensemble predictions from all models.
        
        Args:
            X_test: Test features
            weights: Optional weights for each model (default: equal weights)
            
        Returns:
            Dictionary with ensemble predictions and individual model predictions
        """
        logger.info("Creating ensemble predictions")
        
        if not self.models:
            logger.warning("No models available for ensemble")
            return {}
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        # Get predictions from all models
        model_predictions = {}
        ensemble_probabilities = np.zeros(len(X_test))
        
        for model_name, model in self.models.items():
            try:
                # Get fraud probabilities
                probabilities = model.predict_proba(X_test)[:, 1]
                model_predictions[model_name] = probabilities
                
                # Weighted ensemble
                ensemble_probabilities += weights[model_name] * probabilities
                
            except Exception as e:
                logger.error("Error getting predictions from %s: %s", model_name, str(e))
                continue
        
        # Add ensemble predictions
        model_predictions['ensemble'] = ensemble_probabilities
        
        # Convert to binary predictions
        binary_predictions = {}
        for name, probs in model_predictions.items():
            binary_predictions[f"{name}_binary"] = (probs > 0.5).astype(int)
        
        # Combine all predictions
        all_predictions = {**model_predictions, **binary_predictions}
        
        self.ensemble_predictions = all_predictions
        return all_predictions
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, float]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to optimize ('roc_auc', 'precision', 'recall', 'f1_score')
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run compare_models first.")
            return None, None
        
        best_model = None
        best_value = -1
        
        for model_name, result in self.comparison_results.items():
            if metric == 'roc_auc':
                value = result.roc_auc
            elif metric == 'precision':
                value = result.precision
            elif metric == 'recall':
                value = result.recall
            elif metric == 'f1_score':
                value = result.f1_score
            else:
                logger.error("Unknown metric: %s", metric)
                return None, None
            
            if value > best_value:
                best_value = value
                best_model = model_name
        
        return best_model, best_value
    
    def generate_comparison_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                                 output_dir: str = "reports") -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save reports
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating comprehensive comparison report")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Compare models
        self.compare_models(X_test, y_test)
        
        # Create performance summary
        summary_df = self.create_performance_summary()
        
        # Generate plots
        roc_fig = self.plot_roc_curves(X_test, y_test)
        pr_fig = self.plot_precision_recall_curves(X_test, y_test)
        metrics_fig = self.plot_performance_metrics()
        time_fig = self.plot_inference_time_comparison()
        
        # Save plots
        roc_fig.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches='tight')
        pr_fig.savefig(output_path / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
        metrics_fig.savefig(output_path / "performance_metrics.png", dpi=300, bbox_inches='tight')
        time_fig.savefig(output_path / "inference_time_comparison.png", dpi=300, bbox_inches='tight')
        
        # Save performance summary
        summary_df.to_csv(output_path / "performance_summary.csv", index=False)
        
        # Generate text report
        report_path = output_path / "model_comparison_report.txt"
        self._write_text_report(report_path, summary_df)
        
        logger.info("Comparison report generated in %s", output_dir)
        return str(output_path)
    
    def _write_text_report(self, filepath: Path, summary_df: pd.DataFrame):
        """Write text-based comparison report."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRAUD DETECTION MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report generated: {pd.Timestamp.now()}\n")
            f.write(f"Models compared: {len(self.models)}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            # Best model analysis
            best_roc_model, best_roc = self.get_best_model('roc_auc')
            best_f1_model, best_f1 = self.get_best_model('f1_score')
            fastest_model = min(self.comparison_results.keys(), 
                              key=lambda x: self.comparison_results[x].inference_time_ms)
            fastest_time = self.comparison_results[fastest_model].inference_time_ms
            
            f.write("BEST PERFORMING MODELS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best ROC-AUC: {best_roc_model} ({best_roc:.4f})\n")
            f.write(f"Best F1-Score: {best_f1_model} ({best_f1:.4f})\n")
            f.write(f"Fastest Inference: {fastest_model} ({fastest_time:.2f}ms)\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if best_roc_model == fastest_model:
                f.write("✓ Best performing model is also the fastest\n")
            else:
                f.write("⚠ Consider trade-off between performance and speed\n")
            
            if fastest_time > 100:
                f.write("⚠ All models exceed 100ms inference time target\n")
            else:
                f.write("✓ All models meet inference time requirements\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info("Text report written to %s", filepath)