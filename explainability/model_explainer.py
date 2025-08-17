"""
Model explainability module for fraud detection system.

This module provides SHAP integration, risk insights generation,
and human-readable explanations for fraud decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Explainability features will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class RiskInsight:
    """Container for risk insights and explanations."""
    risk_level: str
    risk_score: float
    top_risk_factors: List[Dict[str, Any]]
    business_explanation: str
    recommendations: List[str]
    confidence: str

@dataclass
class FeatureContribution:
    """Container for feature contribution analysis."""
    feature_name: str
    contribution_value: float
    contribution_percent: float
    risk_direction: str
    business_meaning: str

class ModelExplainer:
    """
    Model explainability and risk insights generator.
    
    This class provides comprehensive explanations for fraud detection
    decisions, including SHAP analysis and business-friendly insights.
    """
    
    def __init__(self):
        """Initialize the model explainer."""
        self.shap_available = SHAP_AVAILABLE
        self.risk_templates = self._load_risk_templates()
        self.feature_descriptions = self._load_feature_descriptions()
        
        logger.info("Model explainer initialized (SHAP: %s)", self.shap_available)
    
    def _load_risk_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load risk explanation templates."""
        return {
            'high_risk': {
                'description': 'This transaction shows multiple high-risk indicators',
                'recommendations': [
                    'Review transaction details carefully',
                    'Consider additional verification steps',
                    'Monitor user account for suspicious activity'
                ]
            },
            'medium_risk': {
                'description': 'This transaction has some concerning characteristics',
                'recommendations': [
                    'Review transaction for unusual patterns',
                    'Consider standard verification procedures'
                ]
            },
            'low_risk': {
                'description': 'This transaction appears legitimate',
                'recommendations': [
                    'Process transaction normally',
                    'No additional verification required'
                ]
            }
        }
    
    def _load_feature_descriptions(self) -> Dict[str, str]:
        """Load human-readable feature descriptions."""
        return {
            'transactions_1h': 'Number of transactions in the last hour',
            'transactions_24h': 'Number of transactions in the last 24 hours',
            'transactions_168h': 'Number of transactions in the last week',
            'total_amount_1h': 'Total amount spent in the last hour',
            'total_amount_24h': 'Total amount spent in the last 24 hours',
            'total_amount_168h': 'Total amount spent in the last week',
            'amount_zscore': 'Transaction amount compared to user\'s typical spending',
            'category_familiarity': 'How familiar the user is with this merchant category',
            'hour_familiarity': 'How familiar the user is with transactions at this hour',
            'ip_transaction_count': 'Number of transactions from this IP address',
            'ip_fraud_rate': 'Historical fraud rate for this IP address',
            'email_domain_risk': 'Risk score for the email domain',
            'country_risk': 'Risk score for the transaction country',
            'location_anomaly': 'Distance from user\'s usual transaction locations',
            'business_hours': 'Whether transaction occurred during business hours',
            'rapid_transaction': 'Whether this transaction follows another very quickly'
        }
    
    def generate_risk_insights(self, features: Dict[str, float], 
                              scoring_result: Dict[str, Any],
                              model_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive risk insights for a transaction.
        
        Args:
            features: Feature dictionary
            scoring_result: Model scoring result
            model_name: Name of the model used
            
        Returns:
            Dictionary with risk insights and explanations
        """
        try:
            risk_score = scoring_result['risk_score']
            confidence = scoring_result['confidence']
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'high_risk'
            elif risk_score >= 0.3:
                risk_level = 'medium_risk'
            else:
                risk_level = 'low_risk'
            
            # Get top risk factors
            top_risk_factors = self._identify_top_risk_factors(features, risk_score)
            
            # Generate business explanation
            business_explanation = self._generate_business_explanation(
                risk_level, top_risk_factors, features
            )
            
            # Get recommendations
            recommendations = self.risk_templates[risk_level]['recommendations']
            
            # Create risk insight
            risk_insight = RiskInsight(
                risk_level=risk_level,
                risk_score=risk_score,
                top_risk_factors=top_risk_factors,
                business_explanation=business_explanation,
                recommendations=recommendations,
                confidence=confidence
            )
            
            return self._format_risk_insights(risk_insight)
            
        except Exception as e:
            logger.error("Error generating risk insights: %s", str(e))
            return self._generate_fallback_insights(scoring_result)
    
    def _identify_top_risk_factors(self, features: Dict[str, float], 
                                  risk_score: float) -> List[Dict[str, Any]]:
        """Identify the top risk factors contributing to the decision."""
        risk_factors = []
        
        # Define risk thresholds for different feature types
        risk_thresholds = {
            'velocity': 0.6,  # High velocity features
            'behavioral': 0.5,  # Behavioral anomalies
            'network': 0.7,    # Network risk features
            'geographic': 0.6,  # Geographic risk features
            'temporal': 0.5    # Temporal risk features
        }
        
        for feature_name, value in features.items():
            risk_contribution = self._calculate_feature_risk_contribution(
                feature_name, value, risk_thresholds
            )
            
            if risk_contribution > 0:
                risk_factors.append({
                    'feature_name': feature_name,
                    'feature_value': value,
                    'risk_contribution': risk_contribution,
                    'description': self.feature_descriptions.get(feature_name, 'Unknown feature'),
                    'risk_type': self._categorize_risk_type(feature_name)
                })
        
        # Sort by risk contribution and return top 5
        risk_factors.sort(key=lambda x: x['risk_contribution'], reverse=True)
        return risk_factors[:5]
    
    def _calculate_feature_risk_contribution(self, feature_name: str, 
                                           value: float,
                                           thresholds: Dict[str, float]) -> float:
        """Calculate risk contribution for a specific feature."""
        risk_contribution = 0.0
        
        # Velocity features (high values indicate risk)
        if 'transactions_' in feature_name or 'amount_' in feature_name:
            if value > 10:  # High velocity threshold
                risk_contribution = min(value / 20, 1.0)  # Normalize to 0-1
        
        # Behavioral features (Z-scores, anomalies)
        elif 'zscore' in feature_name or 'anomaly' in feature_name:
            if abs(value) > 2.0:  # Significant deviation
                risk_contribution = min(abs(value) / 4.0, 1.0)
        
        # Network features (IP reputation, email risk)
        elif 'fraud_rate' in feature_name or 'risk' in feature_name:
            risk_contribution = value  # Direct risk score
        
        # Geographic features (location anomalies)
        elif 'distance' in feature_name or 'anomaly' in feature_name:
            if value > 100:  # 100km threshold
                risk_contribution = min(value / 500, 1.0)
        
        # Temporal features (unusual timing)
        elif 'rapid' in feature_name or 'unusual' in feature_name:
            risk_contribution = value if value > 0 else 0
        
        return risk_contribution
    
    def _categorize_risk_type(self, feature_name: str) -> str:
        """Categorize feature by risk type."""
        if any(x in feature_name for x in ['transactions_', 'amount_', 'velocity']):
            return 'velocity'
        elif any(x in feature_name for x in ['zscore', 'familiarity', 'pattern']):
            return 'behavioral'
        elif any(x in feature_name for x in ['ip_', 'email_', 'domain']):
            return 'network'
        elif any(x in feature_name for x in ['country_', 'location_', 'distance']):
            return 'geographic'
        elif any(x in feature_name for x in ['hour_', 'time_', 'rapid']):
            return 'temporal'
        else:
            return 'other'
    
    def _generate_business_explanation(self, risk_level: str,
                                     top_risk_factors: List[Dict[str, Any]],
                                     features: Dict[str, float]) -> str:
        """Generate human-readable business explanation."""
        template = self.risk_templates[risk_level]['description']
        
        if not top_risk_factors:
            return template
        
        # Add specific risk factor details
        factor_explanations = []
        for factor in top_risk_factors[:3]:  # Top 3 factors
            description = factor['description']
            value = factor['feature_value']
            risk_type = factor['risk_type']
            
            if risk_type == 'velocity':
                factor_explanations.append(
                    f"unusually high {description} ({value:.1f})"
                )
            elif risk_type == 'behavioral':
                factor_explanations.append(
                    f"unusual {description} pattern"
                )
            elif risk_type == 'network':
                factor_explanations.append(
                    f"suspicious {description} activity"
                )
            elif risk_type == 'geographic':
                factor_explanations.append(
                    f"unusual {description} location"
                )
            elif risk_type == 'temporal':
                factor_explanations.append(
                    f"unusual {description} timing"
                )
        
        if factor_explanations:
            explanation = f"{template} Specifically: {', '.join(factor_explanations)}."
        else:
            explanation = template
        
        return explanation
    
    def _format_risk_insights(self, risk_insight: RiskInsight) -> Dict[str, Any]:
        """Format risk insights for API response."""
        return {
            'risk_level': risk_insight.risk_level,
            'risk_score': risk_insight.risk_score,
            'confidence': risk_insight.confidence,
            'top_risk_factors': risk_insight.top_risk_factors,
            'business_explanation': risk_insight.business_explanation,
            'recommendations': risk_insight.recommendations,
            'risk_breakdown': {
                'velocity_risk': self._calculate_category_risk(risk_insight.top_risk_factors, 'velocity'),
                'behavioral_risk': self._calculate_category_risk(risk_insight.top_risk_factors, 'behavioral'),
                'network_risk': self._calculate_category_risk(risk_insight.top_risk_factors, 'network'),
                'geographic_risk': self._calculate_category_risk(risk_insight.top_risk_factors, 'geographic'),
                'temporal_risk': self._calculate_category_risk(risk_insight.top_risk_factors, 'temporal')
            }
        }
    
    def _calculate_category_risk(self, risk_factors: List[Dict[str, Any]], 
                                category: str) -> float:
        """Calculate aggregate risk for a specific category."""
        category_factors = [f for f in risk_factors if f['risk_type'] == category]
        
        if not category_factors:
            return 0.0
        
        # Average risk contribution for this category
        return np.mean([f['risk_contribution'] for f in category_factors])
    
    def _generate_fallback_insights(self, scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback insights when detailed analysis fails."""
        risk_score = scoring_result['risk_score']
        
        if risk_score >= 0.7:
            risk_level = 'high_risk'
        elif risk_score >= 0.3:
            risk_level = 'medium_risk'
        else:
            risk_level = 'low_risk'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': scoring_result.get('confidence', 'unknown'),
            'business_explanation': f"Transaction scored with {risk_level.replace('_', ' ')} level",
            'recommendations': self.risk_templates[risk_level]['recommendations'],
            'top_risk_factors': [],
            'risk_breakdown': {}
        }
    
    def generate_shap_explanation(self, model, features: pd.DataFrame,
                                 feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP-based feature explanations.
        
        Args:
            model: Trained model instance
            features: Feature DataFrame
            feature_names: List of feature names
            
        Returns:
            SHAP explanation data or None if SHAP unavailable
        """
        if not self.shap_available:
            logger.warning("SHAP not available for detailed explanations")
            return None
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, features)
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, features)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features)
            
            # If shap_values is a list, take the first element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Get feature importance
            feature_importance = np.abs(shap_values).mean(0)
            
            # Create feature contribution analysis
            feature_contributions = []
            for i, (feature_name, importance) in enumerate(zip(feature_names, feature_importance)):
                contribution = FeatureContribution(
                    feature_name=feature_name,
                    contribution_value=float(importance),
                    contribution_percent=float(importance / feature_importance.sum() * 100),
                    risk_direction='positive' if importance > 0 else 'negative',
                    business_meaning=self.feature_descriptions.get(feature_name, 'Unknown feature')
                )
                feature_contributions.append(contribution)
            
            # Sort by contribution value
            feature_contributions.sort(key=lambda x: x.contribution_value, reverse=True)
            
            return {
                'shap_values': shap_values.tolist(),
                'feature_contributions': [
                    {
                        'feature_name': fc.feature_name,
                        'contribution_value': fc.contribution_value,
                        'contribution_percent': fc.contribution_percent,
                        'risk_direction': fc.risk_direction,
                        'business_meaning': fc.business_meaning
                    }
                    for fc in feature_contributions
                ],
                'expected_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None
            }
            
        except Exception as e:
            logger.error("Error generating SHAP explanation: %s", str(e))
            return None
    
    def create_explanation_visualization(self, features: Dict[str, float],
                                       risk_insights: Dict[str, Any],
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualization data for risk explanations.
        
        Args:
            features: Feature dictionary
            risk_insights: Risk insights data
            output_path: Path to save visualization (optional)
            
        Returns:
            Visualization data dictionary
        """
        try:
            # Prepare data for visualization
            risk_factors = risk_insights.get('top_risk_factors', [])
            
            if not risk_factors:
                return {'error': 'No risk factors available for visualization'}
            
            # Create waterfall chart data
            waterfall_data = []
            cumulative_risk = 0.0
            
            for factor in risk_factors:
                contribution = factor['risk_contribution']
                waterfall_data.append({
                    'factor': factor['feature_name'],
                    'contribution': contribution,
                    'cumulative': cumulative_risk + contribution,
                    'description': factor['description']
                })
                cumulative_risk += contribution
            
            # Create risk breakdown chart data
            risk_breakdown = risk_insights.get('risk_breakdown', {})
            breakdown_data = [
                {'category': cat.replace('_', ' ').title(), 'risk': risk}
                for cat, risk in risk_breakdown.items()
            ]
            
            visualization_data = {
                'waterfall_chart': {
                    'title': 'Risk Factor Contributions',
                    'data': waterfall_data,
                    'total_risk': risk_insights['risk_score']
                },
                'risk_breakdown': {
                    'title': 'Risk by Category',
                    'data': breakdown_data
                },
                'risk_summary': {
                    'overall_risk': risk_insights['risk_level'],
                    'risk_score': risk_insights['risk_score'],
                    'confidence': risk_insights['confidence']
                }
            }
            
            # Save visualization data if path provided
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(visualization_data, f, indent=2)
                
                logger.info("Visualization data saved to %s", output_path)
            
            return visualization_data
            
        except Exception as e:
            logger.error("Error creating explanation visualization: %s", str(e))
            return {'error': f'Visualization failed: {str(e)}'}
    
    def get_explanation_summary(self, features: Dict[str, float],
                               scoring_result: Dict[str, Any],
                               model_name: str) -> str:
        """
        Get a concise explanation summary for the transaction.
        
        Args:
            features: Feature dictionary
            scoring_result: Model scoring result
            model_name: Name of the model used
            
        Returns:
            Concise explanation string
        """
        try:
            risk_score = scoring_result['risk_score']
            
            if risk_score >= 0.7:
                summary = "HIGH RISK: Multiple suspicious indicators detected"
            elif risk_score >= 0.3:
                summary = "MEDIUM RISK: Some concerning patterns observed"
            else:
                summary = "LOW RISK: Transaction appears legitimate"
            
            # Add top risk factor if available
            top_factors = self._identify_top_risk_factors(features, risk_score)
            if top_factors:
                top_factor = top_factors[0]
                summary += f" - Primary concern: {top_factor['description']}"
            
            return summary
            
        except Exception as e:
            logger.error("Error generating explanation summary: %s", str(e))
            return f"Risk score: {scoring_result.get('risk_score', 'unknown')}"