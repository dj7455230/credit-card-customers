"""
MLOps monitoring and model performance tracking utilities
"""
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    def __init__(self):
        self.performance_history = {}
        self.drift_thresholds = {
            'accuracy': 0.05,  # 5% drop triggers alert
            'precision': 0.05,
            'recall': 0.05,
            'feature_drift': 0.1
        }
        
    def log_prediction(self, model_name: str, prediction: Dict[str, Any], 
                      ground_truth: int = None, features: Dict[str, Any] = None):
        """Log individual prediction for monitoring"""
        timestamp = datetime.now()
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                'predictions': [],
                'performance_metrics': [],
                'drift_scores': [],
                'alerts': []
            }
        
        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'features': features
        }
        
        self.performance_history[model_name]['predictions'].append(log_entry)
        
        # Trigger performance calculation if we have ground truth
        if ground_truth is not None:
            self._update_performance_metrics(model_name)
    
    def _update_performance_metrics(self, model_name: str):
        """Update performance metrics for a model"""
        predictions = self.performance_history[model_name]['predictions']
        
        # Get recent predictions with ground truth
        recent_preds = [p for p in predictions[-100:] if p['ground_truth'] is not None]
        
        if len(recent_preds) < 10:  # Need minimum samples
            return
        
        y_true = [p['ground_truth'] for p in recent_preds]
        y_pred = [p['prediction']['prediction'] for p in recent_preds]
        y_proba = [p['prediction']['probability'] for p in recent_preds]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'timestamp': datetime.now(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.5,
            'sample_size': len(recent_preds)
        }
        
        self.performance_history[model_name]['performance_metrics'].append(metrics)
        
        # Check for performance drift
        self._check_performance_drift(model_name, metrics)
    
    def _check_performance_drift(self, model_name: str, current_metrics: Dict[str, Any]):
        """Check if model performance has drifted significantly"""
        metrics_history = self.performance_history[model_name]['performance_metrics']
        
        if len(metrics_history) < 2:
            return
        
        # Compare with baseline (first recorded metrics)
        baseline = metrics_history[0]
        
        drift_detected = False
        drift_details = {}
        
        for metric in ['accuracy', 'precision', 'recall']:
            baseline_value = baseline[metric]
            current_value = current_metrics[metric]
            drift = baseline_value - current_value
            
            if drift > self.drift_thresholds[metric]:
                drift_detected = True
                drift_details[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'drift': drift,
                    'threshold': self.drift_thresholds[metric]
                }
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now(),
                'type': 'performance_drift',
                'model': model_name,
                'severity': 'high' if any(d['drift'] > 0.1 for d in drift_details.values()) else 'medium',
                'details': drift_details
            }
            
            self.performance_history[model_name]['alerts'].append(alert)
    
    def detect_feature_drift(self, model_name: str, new_features: Dict[str, Any], 
                           reference_features: Dict[str, Any] = None):
        """Detect if input features have drifted from training distribution"""
        
        if reference_features is None:
            # Use historical features as reference
            predictions = self.performance_history.get(model_name, {}).get('predictions', [])
            if not predictions:
                return {'drift_detected': False, 'message': 'No reference data available'}
            
            reference_features = predictions[-100:]  # Last 100 predictions as reference
        
        drift_scores = {}
        drift_detected = False
        
        # Calculate simple statistical drift (in production, use more sophisticated methods)
        for feature, value in new_features.items():
            if feature in reference_features:
                ref_values = [p.get('features', {}).get(feature, 0) for p in reference_features 
                            if p.get('features') and feature in p['features']]
                
                if ref_values:
                    ref_mean = np.mean(ref_values)
                    ref_std = np.std(ref_values) if len(ref_values) > 1 else 1
                    
                    # Z-score based drift detection
                    z_score = abs(value - ref_mean) / ref_std if ref_std > 0 else 0
                    drift_scores[feature] = z_score
                    
                    if z_score > 3:  # 3 sigma rule
                        drift_detected = True
        
        if drift_detected:
            alert = {
                'timestamp': datetime.now(),
                'type': 'feature_drift',
                'model': model_name,
                'severity': 'medium',
                'drift_scores': drift_scores,
                'features_affected': [f for f, score in drift_scores.items() if score > 3]
            }
            
            if model_name not in self.performance_history:
                self.performance_history[model_name] = {'alerts': []}
            
            self.performance_history[model_name]['alerts'].append(alert)
        
        return {
            'drift_detected': drift_detected,
            'drift_scores': drift_scores,
            'threshold': 3.0
        }
    
    def get_model_health(self, model_name: str):
        """Get overall health status of a model"""
        if model_name not in self.performance_history:
            return {'status': 'unknown', 'message': 'No monitoring data available'}
        
        history = self.performance_history[model_name]
        recent_alerts = [a for a in history.get('alerts', []) 
                        if a['timestamp'] > datetime.now() - timedelta(days=7)]
        
        # Determine health status
        if not recent_alerts:
            status = 'healthy'
            message = 'No issues detected in the last 7 days'
        else:
            high_severity_alerts = [a for a in recent_alerts if a.get('severity') == 'high']
            if high_severity_alerts:
                status = 'critical'
                message = f'{len(high_severity_alerts)} critical issues detected'
            else:
                status = 'warning'
                message = f'{len(recent_alerts)} issues detected'
        
        # Get latest performance metrics
        latest_metrics = {}
        if history.get('performance_metrics'):
            latest_metrics = history['performance_metrics'][-1]
        
        return {
            'status': status,
            'message': message,
            'recent_alerts': len(recent_alerts),
            'latest_metrics': latest_metrics,
            'total_predictions': len(history.get('predictions', [])),
            'last_updated': datetime.now()
        }
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report for all models"""
        report = {
            'generated_at': datetime.now(),
            'models': {}
        }
        
        for model_name in self.performance_history:
            health = self.get_model_health(model_name)
            history = self.performance_history[model_name]
            
            report['models'][model_name] = {
                'health_status': health,
                'total_predictions': len(history.get('predictions', [])),
                'performance_metrics_count': len(history.get('performance_metrics', [])),
                'alerts_count': len(history.get('alerts', [])),
                'recent_performance': history.get('performance_metrics', [])[-1] if history.get('performance_metrics') else None
            }
        
        return report

class ExperimentTracker:
    def __init__(self):
        self.experiments = {}
        self.active_experiments = {}
    
    def start_experiment(self, experiment_name: str, description: str = "", 
                        config: Dict[str, Any] = None):
        """Start a new ML experiment"""
        experiment_id = f"exp_{int(datetime.now().timestamp())}"
        
        experiment = {
            'id': experiment_id,
            'name': experiment_name,
            'description': description,
            'config': config or {},
            'start_time': datetime.now(),
            'status': 'running',
            'metrics': [],
            'artifacts': []
        }
        
        self.experiments[experiment_id] = experiment
        self.active_experiments[experiment_name] = experiment_id
        
        return experiment_id
    
    def log_metric(self, experiment_id: str, metric_name: str, value: float, step: int = None):
        """Log a metric for an experiment"""
        if experiment_id in self.experiments:
            metric = {
                'name': metric_name,
                'value': value,
                'step': step,
                'timestamp': datetime.now()
            }
            self.experiments[experiment_id]['metrics'].append(metric)
    
    def log_artifact(self, experiment_id: str, artifact_name: str, artifact_path: str):
        """Log an artifact (file) for an experiment"""
        if experiment_id in self.experiments:
            artifact = {
                'name': artifact_name,
                'path': artifact_path,
                'timestamp': datetime.now()
            }
            self.experiments[experiment_id]['artifacts'].append(artifact)
    
    def end_experiment(self, experiment_id: str, status: str = 'completed'):
        """End an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = status
            self.experiments[experiment_id]['end_time'] = datetime.now()
            
            # Remove from active experiments
            for name, exp_id in list(self.active_experiments.items()):
                if exp_id == experiment_id:
                    del self.active_experiments[name]
    
    def get_experiment_summary(self):
        """Get summary of all experiments"""
        return {
            'total_experiments': len(self.experiments),
            'active_experiments': len(self.active_experiments),
            'experiments': list(self.experiments.values())
        }

# Global monitoring instances
model_monitor = ModelMonitor()
experiment_tracker = ExperimentTracker()