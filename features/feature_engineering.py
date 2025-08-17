"""
Feature engineering module for fraud detection system.

This module implements real-time feature computation optimized for <100ms inference time,
including velocity features, behavioral features, network features, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import re

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Container for computed features."""
    velocity_features: Dict[str, float]
    behavioral_features: Dict[str, float]
    network_features: Dict[str, float]
    geographic_features: Dict[str, float]
    temporal_features: Dict[str, float]
    computed_at: datetime

class VelocityFeatures:
    """Compute velocity-based features for fraud detection."""
    
    def __init__(self, window_hours: List[int] = [1, 24, 168]):
        """
        Initialize velocity feature calculator.
        
        Args:
            window_hours: List of time windows in hours [1h, 24h, 7d]
        """
        self.window_hours = window_hours
        self.transaction_history = defaultdict(list)
    
    def add_transaction(self, user_id: str, amount: float, timestamp: datetime):
        """Add transaction to user history for velocity calculations."""
        self.transaction_history[user_id].append({
            'amount': amount,
            'timestamp': timestamp
        })
        
        # Keep only recent transactions (last 30 days)
        cutoff = timestamp - timedelta(days=30)
        self.transaction_history[user_id] = [
            t for t in self.transaction_history[user_id] 
            if t['timestamp'] > cutoff
        ]
    
    def compute_features(self, user_id: str, current_time: datetime) -> Dict[str, float]:
        """Compute velocity features for a user at current time."""
        features = {}
        user_transactions = self.transaction_history.get(user_id, [])
        
        for window_hours in self.window_hours:
            window_start = current_time - timedelta(hours=window_hours)
            
            # Transactions in window
            window_transactions = [
                t for t in user_transactions 
                if t['timestamp'] >= window_start
            ]
            
            # Basic counts
            features[f'transactions_{window_hours}h'] = len(window_transactions)
            
            if window_transactions:
                # Amount statistics
                amounts = [t['amount'] for t in window_transactions]
                features[f'total_amount_{window_hours}h'] = sum(amounts)
                features[f'avg_amount_{window_hours}h'] = np.mean(amounts)
                features[f'max_amount_{window_hours}h'] = max(amounts)
                features[f'min_amount_{window_hours}h'] = min(amounts)
                
                # Velocity ratios
                if window_hours > 1:
                    prev_window = window_hours // 2
                    prev_start = current_time - timedelta(hours=prev_window)
                    prev_transactions = [
                        t for t in user_transactions 
                        if t['timestamp'] >= prev_start
                    ]
                    
                    if prev_transactions:
                        features[f'velocity_ratio_{window_hours}h'] = (
                            len(window_transactions) / len(prev_transactions)
                        )
                    else:
                        features[f'velocity_ratio_{window_hours}h'] = 0.0
                else:
                    features[f'velocity_ratio_{window_hours}h'] = 0.0
            else:
                # No transactions in window
                features[f'total_amount_{window_hours}h'] = 0.0
                features[f'avg_amount_{window_hours}h'] = 0.0
                features[f'max_amount_{window_hours}h'] = 0.0
                features[f'min_amount_{window_hours}h'] = 0.0
                features[f'velocity_ratio_{window_hours}h'] = 0.0
        
        return features

class BehavioralFeatures:
    """Compute behavioral features for fraud detection."""
    
    def __init__(self):
        """Initialize behavioral feature calculator."""
        self.user_profiles = defaultdict(lambda: {
            'amounts': [],
            'categories': defaultdict(int),
            'hours': defaultdict(int),
            'locations': []
        })
    
    def update_profile(self, user_id: str, amount: float, 
                      category: str, hour: int, lat: float, lng: float):
        """Update user behavioral profile."""
        profile = self.user_profiles[user_id]
        profile['amounts'].append(amount)
        profile['categories'][category] += 1
        profile['hours'][hour] += 1
        profile['locations'].append((lat, lng))
        
        # Keep only recent data (last 100 transactions)
        if len(profile['amounts']) > 100:
            profile['amounts'] = profile['amounts'][-100:]
            profile['locations'] = profile['locations'][-100:]
    
    def compute_features(self, user_id: str, amount: float, 
                        category: str, hour: int, lat: float, lng: float) -> Dict[str, float]:
        """Compute behavioral features for current transaction."""
        features = {}
        profile = self.user_profiles[user_id]
        
        if profile['amounts']:
            # Amount Z-score
            amount_mean = np.mean(profile['amounts'])
            amount_std = np.std(profile['amounts'])
            if amount_std > 0:
                features['amount_zscore'] = (amount - amount_mean) / amount_std
            else:
                features['amount_zscore'] = 0.0
            
            # Amount percentile
            features['amount_percentile'] = np.percentile(profile['amounts'], 95)
            
            # Category familiarity
            total_transactions = sum(profile['categories'].values())
            if total_transactions > 0:
                features['category_familiarity'] = (
                    profile['categories'].get(category, 0) / total_transactions
                )
            else:
                features['category_familiarity'] = 0.0
            
            # Hour familiarity
            total_hour_transactions = sum(profile['hours'].values())
            if total_hour_transactions > 0:
                features['hour_familiarity'] = (
                    profile['hours'].get(hour, 0) / total_hour_transactions
                )
            else:
                features['hour_familiarity'] = 0.0
            
            # Location familiarity (distance from usual locations)
            if profile['locations']:
                distances = [
                    self._calculate_distance(lat, lng, loc_lat, loc_lng)
                    for loc_lat, loc_lng in profile['locations']
                ]
                features['avg_location_distance'] = np.mean(distances)
                features['max_location_distance'] = max(distances)
            else:
                features['avg_location_distance'] = 0.0
                features['max_location_distance'] = 0.0
        else:
            # New user
            features['amount_zscore'] = 0.0
            features['amount_percentile'] = 0.0
            features['category_familiarity'] = 0.0
            features['hour_familiarity'] = 0.0
            features['avg_location_distance'] = 0.0
            features['max_location_distance'] = 0.0
        
        return features
    
    def _calculate_distance(self, lat1: float, lng1: float, 
                          lat2: float, lng2: float) -> float:
        """Calculate approximate distance between two coordinates."""
        # Simplified distance calculation (Haversine would be more accurate)
        lat_diff = abs(lat1 - lat2)
        lng_diff = abs(lng1 - lng2)
        return np.sqrt(lat_diff**2 + lng_diff**2) * 111  # Rough km conversion

class NetworkFeatures:
    """Compute network-related features for fraud detection."""
    
    def __init__(self):
        """Initialize network feature calculator."""
        self.ip_reputation = defaultdict(lambda: {
            'transactions': 0,
            'users': set(),
            'fraud_count': 0,
            'last_seen': None
        })
        self.email_domains = defaultdict(lambda: {
            'transactions': 0,
            'fraud_count': 0,
            'risk_score': 0.0
        })
    
    def update_reputation(self, ip: str, user_id: str, is_fraud: bool):
        """Update IP reputation data."""
        reputation = self.ip_reputation[ip]
        reputation['transactions'] += 1
        reputation['users'].add(user_id)
        if is_fraud:
            reputation['fraud_count'] += 1
        reputation['last_seen'] = datetime.now()
    
    def update_email_domain(self, email: str, is_fraud: bool):
        """Update email domain reputation."""
        domain = self._extract_domain(email)
        domain_data = self.email_domains[domain]
        domain_data['transactions'] += 1
        if is_fraud:
            domain_data['fraud_count'] += 1
        
        # Calculate risk score
        if domain_data['transactions'] > 0:
            domain_data['risk_score'] = domain_data['fraud_count'] / domain_data['transactions']
    
    def compute_features(self, ip: str, email: str) -> Dict[str, float]:
        """Compute network features for current transaction."""
        features = {}
        
        # IP reputation features
        ip_rep = self.ip_reputation[ip]
        features['ip_transaction_count'] = ip_rep['transactions']
        features['ip_unique_users'] = len(ip_rep['users'])
        features['ip_fraud_rate'] = (
            ip_rep['fraud_count'] / max(ip_rep['transactions'], 1)
        )
        
        # IP risk indicators
        features['ip_suspicious_users'] = 1 if ip_rep['unique_users'] > 5 else 0
        features['ip_high_fraud'] = 1 if ip_rep['fraud_rate'] > 0.1 else 0
        
        # Email domain features
        domain = self._extract_domain(email)
        domain_data = self.email_domains[domain]
        features['email_domain_transactions'] = domain_data['transactions']
        features['email_domain_risk'] = domain_data['risk_score']
        
        # Email risk indicators
        features['email_temp_domain'] = 1 if self._is_temporary_domain(domain) else 0
        features['email_disposable'] = 1 if self._is_disposable_domain(domain) else 0
        
        return features
    
    def _extract_domain(self, email: str) -> str:
        """Extract domain from email address."""
        return email.split('@')[-1].lower() if '@' in email else email
    
    def _is_temporary_domain(self, domain: str) -> bool:
        """Check if domain is temporary/throwaway."""
        temp_domains = {
            'temp.com', 'throwaway.com', '10minutemail.com',
            'guerrillamail.com', 'mailinator.com'
        }
        return domain in temp_domains
    
    def _is_disposable_domain(self, domain: str) -> bool:
        """Check if domain is disposable email service."""
        disposable_domains = {
            'yopmail.com', 'tempmail.org', 'sharklasers.com',
            'getairmail.com', 'mailnesia.com'
        }
        return domain in disposable_domains

class GeographicFeatures:
    """Compute geographic features for fraud detection."""
    
    def __init__(self):
        """Initialize geographic feature calculator."""
        self.country_risk_scores = {
            'US': 0.1, 'CA': 0.15, 'UK': 0.2, 'DE': 0.25,
            'FR': 0.3, 'JP': 0.35, 'AU': 0.4, 'BR': 0.6,
            'RU': 0.8, 'NG': 0.9, 'CN': 0.7
        }
        self.user_locations = defaultdict(list)
    
    def update_user_location(self, user_id: str, lat: float, lng: float):
        """Update user's location history."""
        self.user_locations[user_id].append((lat, lng))
        
        # Keep only recent locations (last 20)
        if len(self.user_locations[user_id]) > 20:
            self.user_locations[user_id] = self.user_locations[user_id][-20:]
    
    def compute_features(self, user_id: str, country: str, 
                        lat: float, lng: float) -> Dict[str, float]:
        """Compute geographic features for current transaction."""
        features = {}
        
        # Country risk
        features['country_risk'] = self.country_risk_scores.get(country, 0.5)
        
        # Location anomaly (distance from user's usual locations)
        user_locs = self.user_locations[user_id]
        if user_locs:
            distances = [
                self._calculate_distance(lat, lng, loc_lat, loc_lng)
                for loc_lat, loc_lng in user_locs
            ]
            features['location_anomaly'] = np.mean(distances)
            features['max_location_distance'] = max(distances)
            
            # Location velocity (how quickly user moved)
            if len(user_locs) >= 2:
                recent_lat, recent_lng = user_locs[-1]
                features['location_velocity'] = self._calculate_distance(
                    lat, lng, recent_lat, recent_lng
                )
            else:
                features['location_velocity'] = 0.0
        else:
            features['location_anomaly'] = 0.0
            features['max_location_distance'] = 0.0
            features['location_velocity'] = 0.0
        
        # High-risk country indicator
        features['high_risk_country'] = 1 if features['country_risk'] > 0.7 else 0
        
        return features
    
    def _calculate_distance(self, lat1: float, lng1: float, 
                          lat2: float, lng2: float) -> float:
        """Calculate approximate distance between two coordinates."""
        lat_diff = abs(lat1 - lat2)
        lng_diff = abs(lng1 - lng2)
        return np.sqrt(lat_diff**2 + lng_diff**2) * 111

class TemporalFeatures:
    """Compute temporal features for fraud detection."""
    
    def __init__(self):
        """Initialize temporal feature calculator."""
        self.user_timing = defaultdict(list)
    
    def update_timing(self, user_id: str, timestamp: datetime):
        """Update user's transaction timing history."""
        self.user_timing[user_id].append(timestamp)
        
        # Keep only recent timestamps (last 50)
        if len(self.user_timing[user_id]) > 50:
            self.user_timing[user_id] = self.user_timing[user_id][-50:]
    
    def compute_features(self, user_id: str, timestamp: datetime) -> Dict[str, float]:
        """Compute temporal features for current transaction."""
        features = {}
        
        # Time-based features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['month'] = timestamp.month
        
        # Business hours indicator
        features['business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
        features['weekend'] = 1 if timestamp.weekday() >= 5 else 0
        
        # User timing patterns
        user_times = self.user_timing[user_id]
        if len(user_times) >= 2:
            # Time between transactions
            time_diffs = []
            for i in range(1, len(user_times)):
                diff = (user_times[i] - user_times[i-1]).total_seconds() / 3600  # hours
                time_diffs.append(diff)
            
            if time_diffs:
                features['avg_time_between_txns'] = np.mean(time_diffs)
                features['min_time_between_txns'] = min(time_diffs)
                
                # Rapid transaction indicator
                features['rapid_transaction'] = 1 if features['min_time_between_txns'] < 0.1 else 0
            else:
                features['avg_time_between_txns'] = 0.0
                features['min_time_between_txns'] = 0.0
                features['rapid_transaction'] = 0
        else:
            features['avg_time_between_txns'] = 0.0
            features['min_time_between_txns'] = 0.0
            features['rapid_transaction'] = 0
        
        return features

class FeatureEngineer:
    """
    Main feature engineering pipeline optimized for <100ms inference time.
    
    This class orchestrates all feature computation and ensures
    real-time performance for fraud detection.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.velocity_features = VelocityFeatures()
        self.behavioral_features = BehavioralFeatures()
        self.network_features = NetworkFeatures()
        self.geographic_features = GeographicFeatures()
        self.temporal_features = TemporalFeatures()
        
        logger.info("Feature engineering pipeline initialized")
    
    def compute_features(self, transaction: Dict[str, Any]) -> FeatureSet:
        """
        Compute all features for a transaction in real-time.
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            FeatureSet with all computed features
            
        Performance target: <100ms
        """
        start_time = datetime.now()
        
        try:
            # Extract transaction data
            user_id = transaction['user_id']
            amount = float(transaction['amount'])
            timestamp = pd.to_datetime(transaction['timestamp'])
            ip_address = transaction['ip_address']
            email = transaction['email']
            category = transaction['merchant_category']
            country = transaction['country']
            lat = float(transaction['latitude'])
            lng = float(transaction['longitude'])
            is_fraud = transaction.get('is_fraud', False)
            
            # Compute features in parallel (simulated)
            velocity_features = self.velocity_features.compute_features(user_id, timestamp)
            behavioral_features = self.behavioral_features.compute_features(
                user_id, amount, category, timestamp.hour, lat, lng
            )
            network_features = self.network_features.compute_features(ip_address, email)
            geographic_features = self.geographic_features.compute_features(
                user_id, country, lat, lng
            )
            temporal_features = self.temporal_features.compute_features(user_id, timestamp)
            
            # Update historical data for future computations
            self.velocity_features.add_transaction(user_id, amount, timestamp)
            self.behavioral_features.update_profile(user_id, amount, category, 
                                                 timestamp.hour, lat, lng)
            self.network_features.update_reputation(ip_address, user_id, is_fraud)
            self.network_features.update_email_domain(email, is_fraud)
            self.geographic_features.update_user_location(user_id, lat, lng)
            self.temporal_features.update_timing(user_id, timestamp)
            
            # Create feature set
            feature_set = FeatureSet(
                velocity_features=velocity_features,
                behavioral_features=behavioral_features,
                network_features=network_features,
                geographic_features=geographic_features,
                temporal_features=temporal_features,
                computed_at=datetime.now()
            )
            
            # Performance check
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            if computation_time > 100:
                logger.warning("Feature computation took %.2fms (>100ms target)", 
                             computation_time)
            
            return feature_set
            
        except Exception as e:
            logger.error("Error computing features: %s", str(e))
            # Return empty feature set on error
            return FeatureSet(
                velocity_features={},
                behavioral_features={},
                network_features={},
                geographic_features={},
                temporal_features={},
                computed_at=datetime.now()
            )
    
    def get_feature_vector(self, feature_set: FeatureSet) -> Dict[str, float]:
        """
        Convert feature set to flat feature vector for ML models.
        
        Args:
            feature_set: Computed feature set
            
        Returns:
            Flat dictionary of feature names and values
        """
        features = {}
        
        # Flatten all feature dictionaries
        for feature_dict in [
            feature_set.velocity_features,
            feature_set.behavioral_features,
            feature_set.network_features,
            feature_set.geographic_features,
            feature_set.temporal_features
        ]:
            features.update(feature_dict)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names for model training."""
        # This would be populated based on actual feature computation
        # For now, return a sample set
        return [
            'transactions_1h', 'transactions_24h', 'transactions_168h',
            'total_amount_1h', 'total_amount_24h', 'total_amount_168h',
            'amount_zscore', 'category_familiarity', 'hour_familiarity',
            'ip_transaction_count', 'ip_fraud_rate', 'email_domain_risk',
            'country_risk', 'location_anomaly', 'business_hours',
            'rapid_transaction'
        ]