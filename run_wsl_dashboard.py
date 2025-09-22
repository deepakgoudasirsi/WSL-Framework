#!/usr/bin/env python3
"""
WSL Framework Dashboard Launcher
================================

This script launches the Streamlit dashboard for the Weakly Supervised Learning Framework.
It provides an interactive interface for users to experiment with different WSL strategies,
model architectures, and datasets.

Usage:
    python run_wsl_dashboard.py [basic|advanced]

Arguments:
    basic    - Run the basic dashboard (default)
    advanced - Run the advanced dashboard with additional features
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def main():
    """Main launcher function"""
    print("🧠 WSL Framework Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("wsl_streamlit_app.py"):
        print("❌ Error: wsl_streamlit_app.py not found!")
        print("   Please run this script from the WSL project directory.")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Determine which dashboard to run
    dashboard_type = "basic"
    if len(sys.argv) > 1:
        dashboard_type = sys.argv[1].lower()
    
    if dashboard_type == "advanced":
        app_file = "wsl_streamlit_advanced.py"
        if not os.path.exists(app_file):
            print(f"❌ Error: {app_file} not found!")
            print("   Using basic dashboard instead.")
            app_file = "wsl_streamlit_app.py"
            dashboard_type = "basic"
    else:
        app_file = "wsl_streamlit_app.py"
    
    print(f"\n🚀 Launching {dashboard_type.title()} WSL Dashboard...")
    print(f"   App file: {app_file}")
    print("\n📋 Dashboard Features:")
    
    if dashboard_type == "advanced":
        print("   ✅ Advanced parameter configuration")
        print("   ✅ Realistic training curves")
        print("   ✅ Resource utilization tracking")
        print("   ✅ Comprehensive performance analysis")
        print("   ✅ Advanced visualizations")
    else:
        print("   ✅ Basic experiment configuration")
        print("   ✅ Performance metrics")
        print("   ✅ Training curves")
        print("   ✅ Confusion matrices")
        print("   ✅ Strategy comparison")
    
    print("\n🌐 The dashboard will open in your browser at:")
    print("   http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use the sidebar to configure experiments")
    print("   - Click 'Run Experiment' to start simulations")
    print("   - View results in the main area")
    print("   - Check experiment history for trends")
    
    print("\n" + "=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            "streamlit", "run", app_file,
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("   Try running: streamlit run wsl_streamlit_app.py")

if __name__ == "__main__":
    main() 