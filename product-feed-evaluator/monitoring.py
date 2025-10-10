"""
Monitoring and Error Handling for Product Feed Evaluator

This module provides comprehensive monitoring, error handling, and recovery
mechanisms for large-scale product feed evaluations.
"""

import json
import time
import logging
import traceback
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pandas as pd


class EvaluationMonitor:
    """Monitors evaluation progress and handles errors gracefully."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "evaluation.log"
        self.error_file = self.output_dir / "errors.json"
        
        # Setup logging
        self.logger = logging.getLogger("evaluation_monitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler for persistent logging
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_progress(self, processed: int, total: int, current_url: str = "", message: str = ""):
        """Log progress information."""
        self.logger.info(f"Progress: {processed}/{total} - {current_url} - {message}")
    
    def log_error(self, product_index: int, product_url: str, error: Exception, context: Dict[str, Any] = None):
        """Log detailed error information."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'product_index': product_index,
            'product_url': product_url,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # Log to file
        self.logger.error(f"Error processing product {product_index} ({product_url}): {error}")
        
        # Save to error file
        errors = self.load_errors()
        errors.append(error_info)
        self.save_errors(errors)
    
    def load_errors(self) -> List[Dict[str, Any]]:
        """Load error history."""
        if self.error_file.exists():
            try:
                with open(self.error_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def save_errors(self, errors: List[Dict[str, Any]]):
        """Save error history."""
        with open(self.error_file, 'w') as f:
            json.dump(errors, f, indent=2)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        errors = self.load_errors()
        if not errors:
            return {'total_errors': 0, 'error_types': {}, 'recent_errors': []}
        
        # Count error types
        error_types = {}
        for error in errors:
            error_type = error.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = errors[-10:] if len(errors) > 10 else errors
        
        return {
            'total_errors': len(errors),
            'error_types': error_types,
            'recent_errors': recent_errors
        }
    
    def clear_errors(self):
        """Clear error history."""
        if self.error_file.exists():
            self.error_file.unlink()


class RateLimiter:
    """Handles API rate limiting and backoff strategies."""

    def __init__(self, max_requests_per_minute: int = 50, max_requests_per_hour: int = 1000):
        self.max_rpm = max_requests_per_minute
        self.max_rph = max_requests_per_hour
        self.request_times: List[datetime] = []
        self.hourly_requests = 0
        self.last_hour_reset = datetime.now()
        self._lock = threading.Lock()
        self._cooldown_until: Optional[datetime] = None

    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limits."""
        now = datetime.now()
        with self._lock:
            if self._cooldown_until and now < self._cooldown_until:
                return False

            # Reset hourly counter if needed
            if now - self.last_hour_reset > timedelta(hours=1):
                self.hourly_requests = 0
                self.last_hour_reset = now

            # Check hourly limit
            if self.hourly_requests >= self.max_rph:
                return False

            # Check minute limit
            minute_ago = now - timedelta(minutes=1)
            recent_requests = [t for t in self.request_times if t > minute_ago]

            return len(recent_requests) < self.max_rpm

    def record_request(self):
        """Record that a request was made."""
        now = datetime.now()
        with self._lock:
            self.request_times.append(now)
            self.hourly_requests += 1

            # Clean old request times
            minute_ago = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > minute_ago]

    def get_wait_time(self) -> float:
        """Get the time to wait before making the next request or cooldown expiry."""
        now = datetime.now()
        with self._lock:
            if self._cooldown_until and now < self._cooldown_until:
                return max(0.0, (self._cooldown_until - now).total_seconds())

            if not self.request_times:
                return 0.0

            minute_ago = now - timedelta(minutes=1)
            recent_requests = [t for t in self.request_times if t > minute_ago]

            if len(recent_requests) >= self.max_rpm:
                # Wait until the oldest request is more than a minute old
                oldest_recent = min(recent_requests)
                wait_until = oldest_recent + timedelta(minutes=1)
                return max(0.0, (wait_until - now).total_seconds())

            return 0.0

    def enter_cooldown(self, seconds: float) -> None:
        """Enter a temporary cooldown period (e.g., after a 429)."""
        with self._lock:
            self._cooldown_until = datetime.now() + timedelta(seconds=seconds)


class MemoryMonitor:
    """Monitors memory usage and provides optimization suggestions."""
    
    def __init__(self):
        self.peak_memory = 0
        self.memory_samples = []
    
    def check_memory(self) -> Dict[str, Any]:
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.memory_samples.append(memory_mb)
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            # Keep only last 100 samples
            if len(self.memory_samples) > 100:
                self.memory_samples = self.memory_samples[-100:]
            
            return {
                'current_mb': memory_mb,
                'peak_mb': self.peak_memory,
                'avg_mb': sum(self.memory_samples) / len(self.memory_samples),
                'samples': len(self.memory_samples)
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if self.peak_memory > 1000:  # More than 1GB
            recommendations.append("Consider reducing batch size to lower memory usage")
        
        if self.peak_memory > 2000:  # More than 2GB
            recommendations.append("Memory usage is high - consider processing in smaller chunks")
        
        if len(self.memory_samples) > 50:
            recent_avg = sum(self.memory_samples[-20:]) / 20
            if recent_avg > self.peak_memory * 0.8:
                recommendations.append("Memory usage is consistently high - may indicate memory leak")
        
        return recommendations


class EvaluationHealthChecker:
    """Comprehensive health checking for evaluation processes."""
    
    def __init__(self, monitor: EvaluationMonitor, rate_limiter: RateLimiter, memory_monitor: MemoryMonitor):
        self.monitor = monitor
        self.rate_limiter = rate_limiter
        self.memory_monitor = memory_monitor
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check error rate
        error_summary = self.monitor.get_error_summary()
        if error_summary['total_errors'] > 0:
            error_rate = error_summary['total_errors']
            if error_rate > 10:
                health_status['issues'].append(f"High error rate: {error_rate} errors")
                health_status['overall_status'] = 'degraded'
            elif error_rate > 5:
                health_status['issues'].append(f"Moderate error rate: {error_rate} errors")
        
        # Check rate limiting
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            health_status['issues'].append(f"Rate limit reached, wait {wait_time:.1f}s")
            health_status['overall_status'] = 'rate_limited'
        
        # Check memory usage
        memory_info = self.memory_monitor.check_memory()
        if 'error' not in memory_info:
            if memory_info['current_mb'] > 2000:
                health_status['issues'].append(f"High memory usage: {memory_info['current_mb']:.1f}MB")
                health_status['overall_status'] = 'memory_pressure'
        
        # Get recommendations
        health_status['recommendations'] = self.memory_monitor.get_memory_recommendations()
        
        return health_status
    
    def should_pause_evaluation(self) -> bool:
        """Determine if evaluation should be paused due to health issues."""
        health = self.check_health()
        return health['overall_status'] in ['rate_limited', 'memory_pressure']


def create_monitoring_suite(
    output_dir: str = "output",
    max_requests_per_minute: int = 50,
    max_requests_per_hour: int = 1000,
) -> tuple:
    """Create a complete monitoring suite."""
    monitor = EvaluationMonitor(output_dir)
    rate_limiter = RateLimiter(max_requests_per_minute=max_requests_per_minute, max_requests_per_hour=max_requests_per_hour)
    memory_monitor = MemoryMonitor()
    health_checker = EvaluationHealthChecker(monitor, rate_limiter, memory_monitor)

    return monitor, rate_limiter, memory_monitor, health_checker