"""
Synthetic data generator for fraud detection system.

This module creates realistic transaction data with embedded fraud patterns
inspired by real-world fraud scenarios and Stripe Radar's approach.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import random
import json
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    """Represents a single transaction record."""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    ip_address: str
    email: str
    card_last4: str
    card_brand: str
    merchant_category: str
    country: str
    city: str
    latitude: float
    longitude: float
    device_type: str
    browser: str
    is_fraud: bool
    fraud_type: Optional[str]
    fraud_reason: Optional[str]

@dataclass
class FraudPattern:
    """Represents a fraud attack pattern."""
    pattern_type: str
    description: str
    risk_score: float
    features: Dict[str, any]

@dataclass
class UserProfile:
    """Represents a user's spending profile."""
    user_id: str
    avg_amount: float
    std_amount: float
    preferred_categories: List[str]
    typical_hours: List[int]
    home_country: str
    home_city: str
    home_lat: float
    home_lng: float
    risk_level: str

class FraudDataGenerator:
    """
    Generates synthetic transaction data with realistic fraud patterns.
    
    Inspired by Stripe Radar's approach to understanding fraud patterns
    and creating training data for ML models.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the fraud data generator."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Fraud pattern definitions
        self.fraud_patterns = self._define_fraud_patterns()
        
        # User profiles for realistic behavior
        self.user_profiles = self._generate_user_profiles()
        
        # Merchant categories and risk levels
        self.merchant_categories = self._get_merchant_categories()
        
        # Geographic data
        self.countries = self._get_country_data()
        
        logger.info("FraudDataGenerator initialized with seed %d", seed)
    
    def _define_fraud_patterns(self) -> List[FraudPattern]:
        """Define realistic fraud attack patterns."""
        return [
            FraudPattern(
                pattern_type="card_testing",
                description="High-velocity small amount transactions to test stolen cards",
                risk_score=0.95,
                features={
                    "velocity_1h": "high",
                    "amount_range": "small",
                    "merchant_diversity": "high",
                    "time_spacing": "rapid"
                }
            ),
            FraudPattern(
                pattern_type="stolen_card",
                description="Large transactions from unusual locations",
                risk_score=0.90,
                features={
                    "amount": "high",
                    "location_anomaly": "high",
                    "time_anomaly": "high",
                    "merchant_risk": "high"
                }
            ),
            FraudPattern(
                pattern_type="account_takeover",
                description="Unusual spending patterns for legitimate accounts",
                risk_score=0.85,
                features={
                    "behavior_change": "high",
                    "amount_anomaly": "high",
                    "time_anomaly": "high",
                    "category_change": "high"
                }
            ),
            FraudPattern(
                pattern_type="merchant_fraud",
                description="Collusion between merchants and fraudsters",
                risk_score=0.80,
                features={
                    "merchant_risk": "high",
                    "amount_patterns": "suspicious",
                    "user_merchant_correlation": "high"
                }
            )
        ]
    
    def _generate_user_profiles(self) -> List[UserProfile]:
        """Generate diverse user spending profiles."""
        profiles = []
        
        # Generate 1000 unique users with diverse characteristics
        for i in range(1000):
            user_id = f"user_{i:06d}"
            
            # Spending patterns
            avg_amount = np.random.lognormal(3.5, 0.8)  # $30-200 range
            std_amount = avg_amount * 0.6
            
            # Preferred categories
            categories = random.sample(self._get_merchant_categories(), 
                                    random.randint(2, 5))
            
            # Typical hours (most users shop during business hours)
            typical_hours = random.sample(range(9, 22), random.randint(4, 8))
            
            # Geographic location
            country = random.choice(list(self.countries.keys()))
            city_data = self.countries[country]
            city = random.choice(list(city_data.keys()))
            lat, lng = city_data[city]
            
            # Risk level (most users are low risk)
            risk_levels = ["low"] * 80 + ["medium"] * 15 + ["high"] * 5
            risk_level = random.choice(risk_levels)
            
            profile = UserProfile(
                user_id=user_id,
                avg_amount=avg_amount,
                std_amount=std_amount,
                preferred_categories=categories,
                typical_hours=typical_hours,
                home_country=country,
                home_city=city,
                home_lat=lat,
                home_lng=lng,
                risk_level=risk_level
            )
            profiles.append(profile)
        
        return profiles
    
    def _get_merchant_categories(self) -> List[str]:
        """Get merchant categories with risk levels."""
        return [
            "electronics", "clothing", "food_delivery", "travel",
            "gaming", "subscription", "gift_cards", "digital_goods",
            "jewelry", "luxury_goods", "pharmacy", "automotive",
            "home_goods", "sports", "books", "music", "movies"
        ]
    
    def _get_country_data(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get country and city coordinates."""
        return {
            "US": {
                "New York": (40.7128, -74.0060),
                "Los Angeles": (34.0522, -118.2437),
                "Chicago": (41.8781, -87.6298),
                "Houston": (29.7604, -95.3698),
                "Phoenix": (33.4484, -112.0740)
            },
            "UK": {
                "London": (51.5074, -0.1278),
                "Manchester": (53.4808, -2.2426),
                "Birmingham": (52.4862, -1.8904)
            },
            "CA": {
                "Toronto": (43.6532, -79.3832),
                "Vancouver": (49.2827, -123.1207),
                "Montreal": (45.5017, -73.5673)
            },
            "DE": {
                "Berlin": (52.5200, 13.4050),
                "Munich": (48.1351, 11.5820),
                "Hamburg": (53.5511, 9.9937)
            },
            "FR": {
                "Paris": (48.8566, 2.3522),
                "Marseille": (43.2965, 5.3698),
                "Lyon": (45.7578, 4.8320)
            }
        }
    
    def generate_transactions(self, 
                            days: int = 30,
                            transactions_per_day: int = 10000,
                            fraud_rate: float = 0.001) -> pd.DataFrame:
        """
        Generate synthetic transaction data.
        
        Args:
            days: Number of days to generate data for
            transactions_per_day: Average transactions per day
            fraud_rate: Target fraud rate (0.001 = 0.1%)
        
        Returns:
            DataFrame with transaction records
        """
        logger.info("Generating %d days of transaction data", days)
        
        transactions = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Calculate total transactions and fraud count
        total_transactions = days * transactions_per_day
        fraud_count = int(total_transactions * fraud_rate)
        
        logger.info("Target: %d total transactions, %d fraud cases", 
                   total_transactions, fraud_count)
        
        # Generate legitimate transactions
        legitimate_count = total_transactions - fraud_count
        legitimate_transactions = self._generate_legitimate_transactions(
            start_date, days, legitimate_count
        )
        transactions.extend(legitimate_transactions)
        
        # Generate fraudulent transactions
        fraud_transactions = self._generate_fraudulent_transactions(
            start_date, days, fraud_count
        )
        transactions.extend(fraud_transactions)
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        logger.info("Generated %d transactions (%d legitimate, %d fraud)", 
                   len(df), len(legitimate_transactions), len(fraud_transactions))
        
        return df
    
    def _generate_legitimate_transactions(self, 
                                        start_date: datetime,
                                        days: int,
                                        count: int) -> List[TransactionData]:
        """Generate legitimate transaction data."""
        transactions = []
        
        for i in range(count):
            # Random date within range
            random_days = random.uniform(0, days)
            timestamp = start_date + timedelta(days=random_days)
            
            # Random user
            user_profile = random.choice(self.user_profiles)
            
            # Amount based on user profile
            amount = np.random.normal(user_profile.avg_amount, 
                                   user_profile.std_amount)
            amount = max(0.01, amount)  # Minimum $0.01
            
            # Merchant selection
            merchant_category = random.choice(user_profile.preferred_categories)
            merchant_id = f"merchant_{merchant_category}_{random.randint(1, 100)}"
            
            # Geographic location (usually near user's home)
            if random.random() < 0.8:  # 80% local transactions
                lat = user_profile.home_lat + np.random.normal(0, 0.1)
                lng = user_profile.home_lng + np.random.normal(0, 0.1)
                country = user_profile.home_country
                city = user_profile.home_city
            else:  # 20% travel transactions
                country = random.choice(list(self.countries.keys()))
                city_data = self.countries[country]
                city = random.choice(list(city_data.keys()))
                lat, lng = city_data[city]
            
            # Generate transaction
            transaction = TransactionData(
                transaction_id=f"txn_{i:08d}",
                user_id=user_profile.user_id,
                merchant_id=merchant_id,
                amount=round(amount, 2),
                currency="USD",
                timestamp=timestamp,
                ip_address=self._generate_ip_address(country),
                email=f"user{random.randint(1, 999999)}@example.com",
                card_last4=f"{random.randint(1000, 9999)}",
                card_brand=random.choice(["visa", "mastercard", "amex"]),
                merchant_category=merchant_category,
                country=country,
                city=city,
                latitude=lat,
                longitude=lng,
                device_type=random.choice(["desktop", "mobile", "tablet"]),
                browser=random.choice(["chrome", "safari", "firefox", "edge"]),
                is_fraud=False,
                fraud_type=None,
                fraud_reason=None
            )
            transactions.append(transaction)
        
        return transactions
    
    def _generate_fraudulent_transactions(self,
                                        start_date: datetime,
                                        days: int,
                                        count: int) -> List[TransactionData]:
        """Generate fraudulent transaction data."""
        transactions = []
        
        for i in range(count):
            # Random date within range
            random_days = random.uniform(0, days)
            timestamp = start_date + timedelta(days=random_days)
            
            # Select fraud pattern
            pattern = random.choice(self.fraud_patterns)
            
            if pattern.pattern_type == "card_testing":
                transaction = self._generate_card_testing_fraud(
                    timestamp, pattern, i
                )
            elif pattern.pattern_type == "stolen_card":
                transaction = self._generate_stolen_card_fraud(
                    timestamp, pattern, i
                )
            elif pattern.pattern_type == "account_takeover":
                transaction = self._generate_account_takeover_fraud(
                    timestamp, pattern, i
                )
            else:  # merchant_fraud
                transaction = self._generate_merchant_fraud(
                    timestamp, pattern, i
                )
            
            transactions.append(transaction)
        
        return transactions
    
    def _generate_card_testing_fraud(self, 
                                   timestamp: datetime,
                                   pattern: FraudPattern,
                                   index: int) -> TransactionData:
        """Generate card testing fraud pattern."""
        # High velocity, small amounts
        amount = random.uniform(0.01, 5.00)  # Small amounts
        
        # Random user (stolen identity)
        user_profile = random.choice(self.user_profiles)
        
        # High-risk merchant
        merchant_category = random.choice(["gift_cards", "digital_goods", "gaming"])
        merchant_id = f"merchant_{merchant_category}_high_risk_{random.randint(1, 50)}"
        
        # Unusual location
        country = random.choice(list(self.countries.keys()))
        city_data = self.countries[country]
        city = random.choice(list(city_data.keys()))
        lat, lng = city_data[city]
        
        return TransactionData(
            transaction_id=f"fraud_card_test_{index:08d}",
            user_id=user_profile.user_id,
            merchant_id=merchant_id,
            amount=round(amount, 2),
            currency="USD",
            timestamp=timestamp,
            ip_address=self._generate_ip_address(country),
            email=f"test{random.randint(1, 999999)}@temp.com",
            card_last4=f"{random.randint(1000, 9999)}",
            card_brand=random.choice(["visa", "mastercard"]),
            merchant_category=merchant_category,
            country=country,
            city=city,
            latitude=lat,
            longitude=lng,
            device_type=random.choice(["desktop", "mobile"]),
            browser=random.choice(["chrome", "firefox"]),
            is_fraud=True,
            fraud_type="card_testing",
            fraud_reason="High velocity small amount transactions from unusual location"
        )
    
    def _generate_stolen_card_fraud(self,
                                   timestamp: datetime,
                                   pattern: FraudPattern,
                                   index: int) -> TransactionData:
        """Generate stolen card fraud pattern."""
        # Large amounts
        amount = random.uniform(500.00, 5000.00)
        
        # Random user (stolen identity)
        user_profile = random.choice(self.user_profiles)
        
        # High-value merchant categories
        merchant_category = random.choice(["jewelry", "luxury_goods", "electronics"])
        merchant_id = f"merchant_{merchant_category}_premium_{random.randint(1, 30)}"
        
        # Very unusual location (far from user's home)
        country = random.choice(list(self.countries.keys()))
        city_data = self.countries[country]
        city = random.choice(list(city_data.keys()))
        lat, lng = city_data[city]
        
        return TransactionData(
            transaction_id=f"fraud_stolen_{index:08d}",
            user_id=user_profile.user_id,
            merchant_id=merchant_id,
            amount=round(amount, 2),
            currency="USD",
            timestamp=timestamp,
            ip_address=self._generate_ip_address(country),
            email=f"user{random.randint(1, 999999)}@example.com",
            card_last4=f"{random.randint(1000, 9999)}",
            card_brand=random.choice(["visa", "mastercard", "amex"]),
            merchant_category=merchant_category,
            country=country,
            city=city,
            latitude=lat,
            longitude=lng,
            device_type=random.choice(["desktop", "mobile"]),
            browser=random.choice(["chrome", "safari"]),
            is_fraud=True,
            fraud_type="stolen_card",
            fraud_reason="Large transaction from unusual location"
        )
    
    def _generate_account_takeover_fraud(self,
                                        timestamp: datetime,
                                        pattern: FraudPattern,
                                        index: int) -> TransactionData:
        """Generate account takeover fraud pattern."""
        # Moderate amounts but unusual for user
        amount = random.uniform(100.00, 800.00)
        
        # Random user (legitimate account)
        user_profile = random.choice(self.user_profiles)
        
        # Unusual merchant category for this user
        unusual_categories = [cat for cat in self.merchant_categories 
                            if cat not in user_profile.preferred_categories]
        merchant_category = random.choice(unusual_categories)
        merchant_id = f"merchant_{merchant_category}_unusual_{random.randint(1, 100)}"
        
        # Unusual time (outside typical hours)
        unusual_hours = [h for h in range(24) if h not in user_profile.typical_hours]
        timestamp = timestamp.replace(hour=random.choice(unusual_hours))
        
        # Location near user's home (account takeover, not stolen card)
        lat = user_profile.home_lat + np.random.normal(0, 0.05)
        lng = user_profile.home_lng + np.random.normal(0, 0.05)
        
        return TransactionData(
            transaction_id=f"fraud_takeover_{index:08d}",
            user_id=user_profile.user_id,
            merchant_id=merchant_id,
            amount=round(amount, 2),
            currency="USD",
            timestamp=timestamp,
            ip_address=self._generate_ip_address(user_profile.home_country),
            email=f"user{random.randint(1, 999999)}@example.com",
            card_last4=f"{random.randint(1000, 9999)}",
            card_brand=random.choice(["visa", "mastercard", "amex"]),
            merchant_category=merchant_category,
            country=user_profile.home_country,
            city=user_profile.home_city,
            latitude=lat,
            longitude=lng,
            device_type=random.choice(["desktop", "mobile"]),
            browser=random.choice(["chrome", "safari", "firefox"]),
            is_fraud=True,
            fraud_type="account_takeover",
            fraud_reason="Unusual spending pattern and timing for legitimate account"
        )
    
    def _generate_merchant_fraud(self,
                                 timestamp: datetime,
                                 pattern: FraudPattern,
                                 index: int) -> TransactionData:
        """Generate merchant fraud pattern."""
        # Moderate amounts
        amount = random.uniform(50.00, 300.00)
        
        # Random user
        user_profile = random.choice(self.user_profiles)
        
        # High-risk merchant
        merchant_category = random.choice(["gift_cards", "digital_goods", "subscription"])
        merchant_id = f"merchant_{merchant_category}_suspicious_{random.randint(1, 20)}"
        
        # Location near user's home (merchant fraud, not stolen card)
        lat = user_profile.home_lat + np.random.normal(0, 0.03)
        lng = user_profile.home_lng + np.random.normal(0, 0.03)
        
        return TransactionData(
            transaction_id=f"fraud_merchant_{index:08d}",
            user_id=user_profile.user_id,
            merchant_id=merchant_id,
            amount=round(amount, 2),
            currency="USD",
            timestamp=timestamp,
            ip_address=self._generate_ip_address(user_profile.home_country),
            email=f"user{random.randint(1, 999999)}@example.com",
            card_last4=f"{random.randint(1000, 9999)}",
            card_brand=random.choice(["visa", "mastercard"]),
            merchant_category=merchant_category,
            country=user_profile.home_country,
            city=user_profile.home_city,
            latitude=lat,
            longitude=lng,
            device_type=random.choice(["desktop", "mobile"]),
            browser=random.choice(["chrome", "safari"]),
            is_fraud=True,
            fraud_type="merchant_fraud",
            fraud_reason="Suspicious merchant with unusual transaction patterns"
        )
    
    def _generate_ip_address(self, country: str) -> str:
        """Generate realistic IP address based on country."""
        # Simplified IP generation for demo purposes
        if country == "US":
            return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif country == "UK":
            return f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif country == "CA":
            return f"172.16.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:
            return f"203.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def save_data(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save generated data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / "transactions.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved transactions to %s", csv_path)
        
        # Save as JSON for API testing
        json_path = output_path / "transactions_sample.json"
        sample_data = df.head(100).to_dict('records')
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
        logger.info("Saved sample transactions to %s", json_path)
        
        # Save metadata
        metadata = {
            "total_transactions": len(df),
            "fraud_count": df['is_fraud'].sum(),
            "fraud_rate": df['is_fraud'].mean(),
            "date_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "fraud_types": df[df['is_fraud']]['fraud_type'].value_counts().to_dict(),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to %s", metadata_path)
        
        return {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "metadata_path": str(metadata_path)
        }

def main():
    """Main function to generate and save synthetic data."""
    logger.info("Starting synthetic data generation...")
    
    # Initialize generator
    generator = FraudDataGenerator(seed=42)
    
    # Generate 30 days of data
    df = generator.generate_transactions(
        days=30,
        transactions_per_day=10000,
        fraud_rate=0.001  # 0.1% fraud rate
    )
    
    # Save data
    output_files = generator.save_data(df)
    
    logger.info("Data generation complete!")
    logger.info("Output files: %s", output_files)
    
    # Print summary statistics
    print("\n=== Data Generation Summary ===")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud cases: {df['is_fraud'].sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    if df['is_fraud'].sum() > 0:
        print("\nFraud type distribution:")
        fraud_types = df[df['is_fraud']]['fraud_type'].value_counts()
        for fraud_type, count in fraud_types.items():
            print(f"  {fraud_type}: {count:,}")

if __name__ == "__main__":
    main()