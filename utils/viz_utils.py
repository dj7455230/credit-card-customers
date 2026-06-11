"""
Advanced visualization utilities using Plotly
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class AdvancedVisualizer:
    def __init__(self):
        self.color_palette = {
            'primary': '#8b5cf6',
            'secondary': '#ec4899', 
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#3b82f6'
        }
    
    def create_model_comparison_chart(self, model_results):
        """Create interactive model comparison chart"""
        models = list(model_results.keys())
        probabilities = [model_results[m]['probability'] for m in models]
        confidences = [model_results[m]['confidence'] for m in models]
        
        fig = go.Figure()
        
        # Probability bars
        fig.add_trace(go.Bar(
            name='Churn Probability',
            x=models,
            y=probabilities,
            marker_color=self.color_palette['danger'],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        ))
        
        # Confidence line
        fig.add_trace(go.Scatter(
            name='Model Confidence',
            x=models,
            y=confidences,
            mode='lines+markers',
            line=dict(color=self.color_palette['info'], width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Model Comparison - Churn Predictions',
            xaxis_title='Models',
            yaxis_title='Churn Probability',
            yaxis2=dict(
                title='Confidence',
                overlaying='y',
                side='right'
            ),
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance):
        """Create feature importance visualization"""
        if not feature_importance:
            return go.Figure()
        
        features, importance = zip(*feature_importance)
        
        fig = go.Figure(go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color=self.color_palette['primary'],
            text=[f'{imp:.3f}' for imp in importance],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Top 10 Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_risk_distribution_chart(self, predictions_df):
        """Create risk distribution histogram"""
        if predictions_df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions_df['probability'],
            nbinsx=20,
            marker_color=self.color_palette['warning'],
            opacity=0.7,
            name='Risk Distribution'
        ))
        
        # Add risk level lines
        fig.add_vline(x=0.2, line_dash="dash", line_color=self.color_palette['success'], 
                     annotation_text="Low Risk")
        fig.add_vline(x=0.5, line_dash="dash", line_color=self.color_palette['warning'], 
                     annotation_text="Moderate Risk")
        fig.add_vline(x=0.75, line_dash="dash", line_color=self.color_palette['danger'], 
                     annotation_text="High Risk")
        
        fig.update_layout(
            title='Customer Risk Distribution',
            xaxis_title='Churn Probability',
            yaxis_title='Number of Customers',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_model_performance_radar(self, performance_data):
        """Create radar chart for model performance"""
        if not performance_data:
            return go.Figure()
        
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for model_name, perf in performance_data.items():
            if 'error' not in perf:
                values = [perf.get(metric, 0) for metric in metrics]
                values.append(values[0])  # Close the radar chart
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model_name,
                    line_color=self.color_palette['primary'] if model_name == 'Naive Bayes' else None
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison",
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_prediction_timeline(self, predictions_df):
        """Create timeline of predictions"""
        if predictions_df.empty:
            return go.Figure()
        
        # Convert timestamp to datetime
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Group by date and calculate metrics
        daily_stats = predictions_df.groupby(predictions_df['timestamp'].dt.date).agg({
            'prediction': ['count', 'sum', 'mean'],
            'probability': 'mean'
        }).round(3)
        
        daily_stats.columns = ['total_predictions', 'churn_count', 'churn_rate', 'avg_probability']
        daily_stats = daily_stats.reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Predictions', 'Average Churn Risk'),
            vertical_spacing=0.1
        )
        
        # Daily predictions
        fig.add_trace(
            go.Scatter(
                x=daily_stats['timestamp'],
                y=daily_stats['total_predictions'],
                mode='lines+markers',
                name='Total Predictions',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Average risk
        fig.add_trace(
            go.Scatter(
                x=daily_stats['timestamp'],
                y=daily_stats['avg_probability'],
                mode='lines+markers',
                name='Avg Churn Risk',
                line=dict(color=self.color_palette['danger'])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Prediction Analytics Over Time',
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def create_customer_segmentation_chart(self, customer_data):
        """Create customer segmentation visualization"""
        # Simulate customer segments based on risk and value
        np.random.seed(42)
        n_customers = len(customer_data) if customer_data else 100
        
        segments = []
        for i in range(n_customers):
            risk = np.random.uniform(0, 1)
            value = np.random.uniform(1000, 50000)
            
            if risk < 0.3 and value > 20000:
                segment = 'Champions'
                color = self.color_palette['success']
            elif risk < 0.3 and value > 10000:
                segment = 'Loyal Customers'
                color = self.color_palette['info']
            elif risk > 0.7 and value > 15000:
                segment = 'At Risk'
                color = self.color_palette['warning']
            elif risk > 0.7:
                segment = 'Cannot Lose Them'
                color = self.color_palette['danger']
            else:
                segment = 'Potential Loyalists'
                color = self.color_palette['primary']
            
            segments.append({
                'risk': risk,
                'value': value,
                'segment': segment,
                'color': color
            })
        
        df = pd.DataFrame(segments)
        
        fig = px.scatter(
            df, x='value', y='risk',
            color='segment',
            title='Customer Segmentation Analysis',
            labels={'value': 'Customer Value ($)', 'risk': 'Churn Risk'},
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_shap_waterfall(self, shap_values, feature_names, base_value, prediction_value):
        """Create SHAP waterfall chart"""
        if shap_values is None:
            return go.Figure()
        
        # Simplified waterfall chart
        fig = go.Figure()
        
        # This is a simplified version - in practice, you'd use shap.plots.waterfall
        fig.add_trace(go.Bar(
            x=feature_names[:10],  # Top 10 features
            y=shap_values[:10] if len(shap_values) >= 10 else shap_values,
            marker_color=[self.color_palette['success'] if v > 0 else self.color_palette['danger'] 
                         for v in shap_values[:10]],
            name='SHAP Values'
        ))
        
        fig.update_layout(
            title='SHAP Feature Contributions',
            xaxis_title='Features',
            yaxis_title='SHAP Value',
            template='plotly_dark',
            height=400
        )
        
        return fig

# Global visualizer instance
visualizer = AdvancedVisualizer()