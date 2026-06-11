#!/usr/bin/env python3
"""
🚀 Advanced Churn Prediction Platform Launcher
Easy setup and launch script for the hackathon-winning ML platform
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🚀 ADVANCED CHURN PREDICTION PLATFORM 🚀                    ║
║                                                                  ║
║     Hackathon-Winning ML-Powered Customer Retention System      ║
║                                                                  ║
║     ✨ 15 Enterprise Features | 4 ML Models | Production Ready   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'gradio', 'pandas', 'numpy', 'scikit-learn', 
        'plotly', 'fastapi', 'uvicorn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected!")
        print("Run this command to install them:")
        print(f"pip install {' '.join(missing_packages)}")
        
        choice = input("\nWould you like to install them now? (y/n): ")
        if choice.lower() == 'y':
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    *missing_packages
                ])
                print("✅ All packages installed successfully!")
            except subprocess.CalledProcessError:
                print("❌ Installation failed. Please install manually.")
                return False
        else:
            return False
    
    return True

def setup_demo_data():
    """Setup demo data if not exists"""
    print("\n📊 Setting up demo data...")
    
    demo_files = [
        'naive.pkl', 'sample_customers.csv', 'demo_config.json'
    ]
    
    missing_files = [f for f in demo_files if not os.path.exists(f)]
    
    if missing_files:
        print("  ⚠️  Demo data not found. Creating...")
        try:
            subprocess.check_call([sys.executable, "create_demo_data.py"])
            print("  ✅ Demo data created successfully!")
        except subprocess.CalledProcessError:
            print("  ❌ Failed to create demo data")
            return False
    else:
        print("  ✅ Demo data already exists")
    
    return True

def launch_application(mode='advanced'):
    """Launch the application"""
    print(f"\n🚀 Launching {mode} application...")
    
    app_file = 'app_advanced.py' if mode == 'advanced' else 'app.py'
    
    if not os.path.exists(app_file):
        print(f"❌ {app_file} not found!")
        return False
    
    print(f"  📱 Starting {app_file}...")
    print("  🌐 The app will open in your browser automatically")
    print("  ⏹️  Press Ctrl+C to stop the application")
    print("\n" + "="*60)
    
    try:
        subprocess.run([sys.executable, app_file])
    except KeyboardInterrupt:
        print("\n\n⏹️  Application stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        return False
    
    return True

def launch_api():
    """Launch the API server"""
    print("\n🔌 Launching API server...")
    
    if not os.path.exists('api.py'):
        print("❌ api.py not found!")
        return False
    
    print("  🌐 API will be available at: http://127.0.0.1:8000")
    print("  📚 API docs at: http://127.0.0.1:8000/docs")
    print("  ⏹️  Press Ctrl+C to stop the API server")
    print("\n" + "="*60)
    
    try:
        subprocess.run([sys.executable, "api.py"])
    except KeyboardInterrupt:
        print("\n\n⏹️  API server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error launching API: {e}")
        return False
    
    return True

def show_menu():
    """Show main menu"""
    menu = """
🎯 What would you like to do?

1. 🚀 Launch Advanced Platform (Full Features)
2. 🎯 Launch Simple Platform (Basic Interface)  
3. 🔌 Launch API Server (FastAPI Endpoints)
4. 📊 Setup Demo Data Only
5. 🏗️  Create Docker Deployment
6. ❓ Show Help & Documentation
7. 🚪 Exit

Choose an option (1-7): """
    
    return input(menu)

def create_docker_deployment():
    """Create Docker deployment files"""
    print("\n🐳 Creating Docker deployment...")
    
    try:
        # Import deployment utilities
        sys.path.append('utils')
        from deployment_utils import deployment_manager
        
        # Create Docker configuration
        docker_files = deployment_manager.create_docker_config()
        
        print("  ✅ Created Docker files:")
        for file_type, filename in docker_files.items():
            print(f"    📄 {filename}")
        
        print("\n🚀 To deploy with Docker:")
        print("  docker-compose up -d --build")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating Docker files: {e}")
        return False

def show_help():
    """Show help and documentation"""
    help_text = """
📚 ADVANCED CHURN PREDICTION PLATFORM - HELP

🎯 FEATURES:
  • Multi-Model ML Prediction (4 algorithms)
  • Real-time Analytics Dashboard  
  • A/B Testing Framework
  • Explainable AI with SHAP
  • AI-Powered Business Insights
  • Automated PDF Reports
  • Email Alert System
  • REST API Endpoints
  • Production Deployment Ready

🚀 QUICK START:
  1. Run this launcher: python launch.py
  2. Choose option 1 for full platform
  3. Open browser to http://127.0.0.1:7863
  4. Explore all tabs and features

🔗 IMPORTANT URLS:
  • Main App: http://127.0.0.1:7863
  • API Docs: http://127.0.0.1:8000/docs
  • API Health: http://127.0.0.1:8000

📁 KEY FILES:
  • app_advanced.py - Full-featured interface
  • app.py - Simple interface
  • api.py - FastAPI server
  • create_demo_data.py - Demo data generator
  • utils/ - All utility modules

🎯 HACKATHON TIPS:
  • Demo the real-time features first
  • Show the multi-model comparison
  • Highlight the AI insights generation
  • Explain the production readiness

📞 SUPPORT:
  • Check README.md for full documentation
  • All code is well-commented
  • Modular design for easy customization

Press Enter to continue...
"""
    
    print(help_text)
    input()

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Launch advanced platform
            if setup_demo_data():
                launch_application('advanced')
            
        elif choice == '2':
            # Launch simple platform
            if setup_demo_data():
                launch_application('simple')
            
        elif choice == '3':
            # Launch API server
            if setup_demo_data():
                launch_api()
            
        elif choice == '4':
            # Setup demo data only
            setup_demo_data()
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            # Create Docker deployment
            create_docker_deployment()
            input("\nPress Enter to continue...")
            
        elif choice == '6':
            # Show help
            show_help()
            
        elif choice == '7':
            # Exit
            print("\n👋 Thanks for using the Advanced Churn Prediction Platform!")
            print("   🌟 Star us on GitHub if you found this helpful!")
            sys.exit(0)
            
        else:
            print("\n❌ Invalid choice. Please select 1-7.")
            time.sleep(1)

if __name__ == "__main__":
    main()