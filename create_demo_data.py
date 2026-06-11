"""
Script to create demo data for the advanced churn prediction system
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta
import random

# Add utils to path
sys.path.append('utils')

def create_sample_models():
    """Create sample model files for demo"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    
    # Try to import optional packages
    try:
        import xgboost as xgb
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("⚠️  XGBoost not available, using RandomForest as substitute")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 40
    
    X = np.random.randn(n_samples, n_features)
    # Create realistic churn pattern
    churn_prob = 1 / (1 + np.exp(-(X[:, 0] * 0.5 + X[:, 1] * -0.3 + X[:, 2] * 0.4 + np.random.normal(0, 0.1, n_samples))))
    y = np.random.binomial(1, churn_prob)
    
    # Feature names
    feature_names = [
        'CLIENTNUM', 'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'Gender_F', 'Gender_M',
        'Education_Level_College', 'Education_Level_Doctorate', 'Education_Level_Graduate',
        'Education_Level_High School', 'Education_Level_Post-Graduate',
        'Education_Level_Uneducated', 'Education_Level_Unknown',
        'Marital_Status_Divorced', 'Marital_Status_Married',
        'Marital_Status_Single', 'Marital_Status_Unknown',
        'Income_Category_$120K +', 'Income_Category_$40K - $60K',
        'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
        'Income_Category_Less than $40K', 'Income_Category_Unknown',
        'Card_Category_Blue', 'Card_Category_Gold',
        'Card_Category_Platinum', 'Card_Category_Silver'
    ]
    
    # Train models
    models = {}
    
    # Naive Bayes (this should be your existing model)
    if not os.path.exists('naive.pkl'):
        nb_model = GaussianNB()
        nb_model.fit(X, y)
        nb_model.feature_names_in_ = np.array(feature_names)
        joblib.dump(nb_model, 'naive.pkl')
        models['naive_bayes'] = nb_model
        print("✅ Created naive.pkl")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_model.feature_names_in_ = np.array(feature_names)
    joblib.dump(rf_model, 'random_forest.pkl')
    models['random_forest'] = rf_model
    print("✅ Created random_forest.pkl")
    
    # XGBoost (if available)
    if has_xgboost:
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X, y)
        xgb_model.feature_names_in_ = np.array(feature_names)
        joblib.dump(xgb_model, 'xgboost.pkl')
        models['xgboost'] = xgb_model
        print("✅ Created xgboost.pkl")
    else:
        # Create another RandomForest as substitute
        xgb_substitute = RandomForestClassifier(n_estimators=150, random_state=123)
        xgb_substitute.fit(X, y)
        xgb_substitute.feature_names_in_ = np.array(feature_names)
        joblib.dump(xgb_substitute, 'xgboost_substitute.pkl')
        models['xgboost'] = xgb_substitute
        print("✅ Created xgboost_substitute.pkl (RandomForest)")
    
    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    nn_model.fit(X, y)
    nn_model.feature_names_in_ = np.array(feature_names)
    joblib.dump(nn_model, 'neural_network.pkl')
    models['neural_network'] = nn_model
    print("✅ Created neural_network.pkl")
    
    return models

def create_sample_dataset():
    """Create sample customer dataset for batch prediction"""
    np.random.seed(42)
    n_customers = 500
    
    # Generate realistic customer data
    data = []
    for i in range(n_customers):
        customer = {
            'CLIENTNUM': 700000000 + i,
            'Customer_Age': np.random.randint(18, 80),
            'Dependent_count': np.random.randint(0, 6),
            'Months_on_book': np.random.randint(12, 60),
            'Total_Relationship_Count': np.random.randint(1, 7),
            'Months_Inactive_12_mon': np.random.randint(0, 7),
            'Contacts_Count_12_mon': np.random.randint(0, 8),
            'Credit_Limit': np.random.uniform(1000, 30000),
            'Total_Revolving_Bal': np.random.uniform(0, 5000),
            'Avg_Open_To_Buy': np.random.uniform(500, 25000),
            'Total_Amt_Chng_Q4_Q1': np.random.uniform(0.3, 1.5),
            'Total_Trans_Amt': np.random.uniform(500, 15000),
            'Total_Trans_Ct': np.random.randint(10, 150),
            'Total_Ct_Chng_Q4_Q1': np.random.uniform(0.3, 1.5),
            'Avg_Utilization_Ratio': np.random.uniform(0, 1),
        }
        
        # NB classifier scores
        customer['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'] = np.random.uniform(0.8, 1.0)
        customer['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'] = np.random.uniform(0, 0.2)
        
        # One-hot encoded features
        # Gender
        gender_f = np.random.choice([0, 1])
        customer['Gender_F'] = gender_f
        customer['Gender_M'] = 1 - gender_f
        
        # Education (one-hot)
        education_categories = ['College', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate', 'Uneducated', 'Unknown']
        edu_choice = np.random.choice(education_categories)
        for cat in education_categories:
            customer[f'Education_Level_{cat}'] = 1 if cat == edu_choice else 0
        
        # Marital Status
        marital_categories = ['Divorced', 'Married', 'Single', 'Unknown']
        mar_choice = np.random.choice(marital_categories)
        for cat in marital_categories:
            customer[f'Marital_Status_{cat}'] = 1 if cat == mar_choice else 0
        
        # Income Category
        income_categories = ['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'Less than $40K', 'Unknown']
        inc_choice = np.random.choice(income_categories)
        for cat in income_categories:
            customer[f'Income_Category_{cat}'] = 1 if cat == inc_choice else 0
        
        # Card Category
        card_categories = ['Blue', 'Gold', 'Platinum', 'Silver']
        card_choice = np.random.choice(card_categories)
        for cat in card_categories:
            customer[f'Card_Category_{cat}'] = 1 if cat == card_choice else 0
        
        data.append(customer)
    
    df = pd.DataFrame(data)
    df.to_csv('sample_customers.csv', index=False)
    print(f"✅ Created sample_customers.csv with {n_customers} customers")
    
    return df

def create_demo_database():
    """Create demo database with sample prediction history"""
    try:
        from db_utils import db_manager
        
        # Generate sample prediction history
        customers = [f"CUST_{i:06d}" for i in range(1000, 2000)]
        models = ['Naive Bayes', 'Random Forest', 'XGBoost', 'Neural Network']
        
        for i in range(500):  # Create 500 sample predictions
            customer_id = random.choice(customers)
            model_used = random.choice(models)
            
            # Generate realistic prediction
            prediction = random.choice([0, 1])
            probability = random.uniform(0, 1)
            confidence = random.uniform(0.7, 0.95)
            
            # Generate sample input features
            input_features = {
                'Customer_Age': random.randint(18, 80),
                'Credit_Limit': random.uniform(1000, 30000),
                'Total_Trans_Ct': random.randint(10, 150),
                'Months_Inactive_12_mon': random.randint(0, 6)
            }
            
            db_manager.save_prediction(
                customer_id=customer_id,
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                model_used=model_used,
                input_features=input_features
            )
        
        print("✅ Created demo database with sample predictions")
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")

def create_reports_directory():
    """Create reports directory with sample reports"""
    os.makedirs('reports', exist_ok=True)
    
    # Create a sample report file
    sample_report_html = """
    <html>
    <head><title>Sample Churn Analysis Report</title></head>
    <body>
        <h1>Customer Churn Analysis Report</h1>
        <p>Generated: {}</p>
        <h2>Executive Summary</h2>
        <p>This is a sample report demonstrating the reporting capabilities.</p>
        <h2>Key Metrics</h2>
        <ul>
            <li>Total Customers Analyzed: 500</li>
            <li>Predicted Churn Rate: 23.4%</li>
            <li>High Risk Customers: 67</li>
            <li>Model Accuracy: 87.5%</li>
        </ul>
    </body>
    </html>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open('reports/sample_report.html', 'w') as f:
        f.write(sample_report_html)
    
    print("✅ Created reports directory with sample report")

def setup_complete_demo():
    """Set up everything needed for the demo"""
    print("🚀 Setting up Advanced Churn Prediction Demo...")
    print("=" * 60)
    
    # Create directories
    os.makedirs('utils', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create models
    print("📊 Creating sample ML models...")
    models = create_sample_models()
    
    # Create sample dataset
    print("📁 Creating sample dataset...")
    create_sample_dataset()
    
    # Create demo database
    print("🗄️ Setting up demo database...")
    create_demo_database()
    
    # Create reports directory
    print("📊 Setting up reports...")
    create_reports_directory()
    
    # Create a demo configuration file
    demo_config = {
        "app_name": "Advanced Churn Predictor",
        "version": "2.0.0",
        "models_available": list(models.keys()) if models else ["naive_bayes"],
        "features": {
            "multi_model_prediction": True,
            "real_time_analytics": True,
            "ab_testing": True,
            "explainable_ai": True,
            "automated_reporting": True,
            "email_alerts": True,
            "api_endpoints": True,
            "model_monitoring": True,
            "deployment_ready": True
        },
        "demo_data": {
            "sample_customers": 500,
            "prediction_history": 500,
            "models_trained": len(models) if models else 1
        },
        "setup_completed": datetime.now().isoformat()
    }
    
    with open('demo_config.json', 'w') as f:
        import json
        json.dump(demo_config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Demo setup completed successfully!")
    print("\n📋 What's been created:")
    print("   • Multiple ML models (Naive Bayes, Random Forest, XGBoost, Neural Network)")
    print("   • Sample customer dataset (500 customers)")
    print("   • Demo database with prediction history")
    print("   • Reports directory with templates")
    print("   • Configuration files")
    print("\n🚀 You can now run:")
    print("   python app_advanced.py")
    print("\n🌐 Then open: http://127.0.0.1:7863")
    print("=" * 60)

if __name__ == "__main__":
    setup_complete_demo()