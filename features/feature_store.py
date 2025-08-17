"""
Feature store module for fraud detection system.

This module provides caching and storage for computed features to optimize
real-time inference performance and reduce redundant computations.
"""

import redis
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)

class FeatureStore(ABC):
    """Abstract base class for feature stores."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get features from store."""
        pass
    
    @abstractmethod
    def set(self, key: str, features: Dict[str, Any], ttl: int = 3600):
        """Set features in store with TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete features from store."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in store."""
        pass

class RedisFeatureStore(FeatureStore):
    """Redis-based feature store for production use."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None):
        """
        Initialize Redis feature store.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # Keep binary for pickle
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis feature store connected successfully")
            
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", str(e))
            raise
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get features from Redis store."""
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("Error getting features from Redis: %s", str(e))
            return None
    
    def set(self, key: str, features: Dict[str, Any], ttl: int = 3600):
        """Set features in Redis store with TTL."""
        try:
            data = pickle.dumps(features)
            self.redis_client.setex(key, ttl, data)
            logger.debug("Stored features in Redis with key: %s, TTL: %d", key, ttl)
        except Exception as e:
            logger.error("Error storing features in Redis: %s", str(e))
    
    def delete(self, key: str):
        """Delete features from Redis store."""
        try:
            self.redis_client.delete(key)
            logger.debug("Deleted features from Redis with key: %s", key)
        except Exception as e:
            logger.error("Error deleting features from Redis: %s", str(e))
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis store."""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error("Error checking key existence in Redis: %s", str(e))
            return False
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection: %s", str(e))

class InMemoryFeatureStore(FeatureStore):
    """In-memory feature store for development and testing."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize in-memory feature store.
        
        Args:
            max_size: Maximum number of features to store
        """
        self.store = {}
        self.max_size = max_size
        self.access_times = {}
        logger.info("In-memory feature store initialized with max size: %d", max_size)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get features from in-memory store."""
        if key in self.store:
            # Update access time for LRU
            self.access_times[key] = datetime.now()
            return self.store[key]
        return None
    
    def set(self, key: str, features: Dict[str, Any], ttl: int = 3600):
        """Set features in in-memory store with TTL."""
        # Add expiration time
        features_with_expiry = {
            'features': features,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        # Check if store is full
        if len(self.store) >= self.max_size:
            self._evict_lru()
        
        self.store[key] = features_with_expiry
        self.access_times[key] = datetime.now()
        
        logger.debug("Stored features in memory with key: %s, TTL: %d", key, ttl)
    
    def delete(self, key: str):
        """Delete features from in-memory store."""
        if key in self.store:
            del self.store[key]
            del self.access_times[key]
            logger.debug("Deleted features from memory with key: %s", key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in in-memory store."""
        if key in self.store:
            # Check if expired
            if datetime.now() > self.store[key]['expires_at']:
                self.delete(key)
                return False
            return True
        return False
    
    def _evict_lru(self):
        """Evict least recently used features when store is full."""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        
        self.delete(lru_key)
        logger.debug("Evicted LRU features to make space")
    
    def cleanup_expired(self):
        """Clean up expired features."""
        current_time = datetime.now()
        expired_keys = [
            key for key, data in self.store.items()
            if current_time > data['expires_at']
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.debug("Cleaned up %d expired features", len(expired_keys))

class FeatureStoreManager:
    """
    Manager for feature stores with fallback and optimization.
    
    This class provides a unified interface to multiple feature stores
    with intelligent caching and fallback strategies.
    """
    
    def __init__(self, primary_store: FeatureStore, 
                 fallback_store: Optional[FeatureStore] = None):
        """
        Initialize feature store manager.
        
        Args:
            primary_store: Primary feature store (e.g., Redis)
            fallback_store: Fallback store (e.g., in-memory)
        """
        self.primary_store = primary_store
        self.fallback_store = fallback_store
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("Feature store manager initialized")
    
    def get_features(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get features with intelligent caching and fallback.
        
        Args:
            key: Feature key
            
        Returns:
            Features dictionary or None if not found
        """
        # Try primary store first
        try:
            features = self.primary_store.get(key)
            if features:
                self.cache_hits += 1
                logger.debug("Cache hit in primary store for key: %s", key)
                return features
        except Exception as e:
            logger.warning("Primary store error, trying fallback: %s", str(e))
        
        # Try fallback store
        if self.fallback_store:
            try:
                features = self.fallback_store.get(key)
                if features:
                    self.cache_hits += 1
                    logger.debug("Cache hit in fallback store for key: %s", key)
                    return features
            except Exception as e:
                logger.warning("Fallback store error: %s", str(e))
        
        self.cache_misses += 1
        logger.debug("Cache miss for key: %s", key)
        return None
    
    def store_features(self, key: str, features: Dict[str, Any], 
                      ttl: int = 3600, use_fallback: bool = True):
        """
        Store features in appropriate stores.
        
        Args:
            key: Feature key
            features: Features to store
            ttl: Time to live in seconds
            use_fallback: Whether to also store in fallback store
        """
        # Store in primary store
        try:
            self.primary_store.set(key, features, ttl)
            logger.debug("Stored features in primary store with key: %s", key)
        except Exception as e:
            logger.error("Error storing in primary store: %s", str(e))
        
        # Store in fallback store if requested
        if use_fallback and self.fallback_store:
            try:
                self.fallback_store.set(key, features, ttl)
                logger.debug("Stored features in fallback store with key: %s", key)
            except Exception as e:
                logger.warning("Error storing in fallback store: %s", str(e))
    
    def generate_key(self, transaction_data: Dict[str, Any]) -> str:
        """
        Generate cache key for transaction data.
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Cache key string
        """
        # Create deterministic key from transaction data
        key_data = {
            'user_id': transaction_data.get('user_id'),
            'ip_address': transaction_data.get('ip_address'),
            'timestamp': transaction_data.get('timestamp'),
            'amount': transaction_data.get('amount')
        }
        
        # Sort for deterministic ordering
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Create hash for consistent key length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"features:{key_hash}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def clear_cache(self):
        """Clear all cached features."""
        try:
            # Clear primary store
            if hasattr(self.primary_store, 'clear'):
                self.primary_store.clear()
            else:
                logger.warning("Primary store does not support clear operation")
            
            # Clear fallback store
            if self.fallback_store and hasattr(self.fallback_store, 'clear'):
                self.fallback_store.clear()
            
            # Reset statistics
            self.cache_hits = 0
            self.cache_misses = 0
            
            logger.info("Feature cache cleared")
            
        except Exception as e:
            logger.error("Error clearing cache: %s", str(e))
    
    def close(self):
        """Close all feature stores."""
        try:
            if hasattr(self.primary_store, 'close'):
                self.primary_store.close()
            
            if self.fallback_store and hasattr(self.fallback_store, 'close'):
                self.fallback_store.close()
            
            logger.info("All feature stores closed")
            
        except Exception as e:
            logger.error("Error closing feature stores: %s", str(e))

def create_feature_store(store_type: str = 'redis', **kwargs) -> FeatureStore:
    """
    Factory function to create feature stores.
    
    Args:
        store_type: Type of store ('redis' or 'memory')
        **kwargs: Store-specific configuration
        
    Returns:
        Configured feature store instance
    """
    if store_type.lower() == 'redis':
        return RedisFeatureStore(**kwargs)
    elif store_type.lower() == 'memory':
        return InMemoryFeatureStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")

def create_feature_store_manager(primary_type: str = 'redis',
                               fallback_type: str = 'memory',
                               **kwargs) -> FeatureStoreManager:
    """
    Factory function to create feature store manager.
    
    Args:
        primary_type: Primary store type
        fallback_type: Fallback store type
        **kwargs: Store-specific configuration
        
    Returns:
        Configured feature store manager
    """
    primary_store = create_feature_store(primary_type, **kwargs)
    fallback_store = create_feature_store(fallback_type, **kwargs)
    
    return FeatureStoreManager(primary_store, fallback_store)