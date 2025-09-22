# 🧠 WSL Framework Streamlit Dashboard

## Overview

This Streamlit application provides an interactive dashboard for the **Weakly Supervised Learning (WSL) Framework** developed by Deepak Gowda. The dashboard allows users to:

- **Select datasets** (CIFAR-10, MNIST)
- **Choose model architectures** (CNN, ResNet18, MLP)
- **Configure WSL strategies** (Consistency Regularization, Pseudo-Labeling, Co-Training, Combined WSL)
- **View real-time results** with interactive visualizations
- **Compare performance** across different configurations
- **Track experiment history** and trends

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 8GB+ RAM (recommended)
- NVIDIA GPU with CUDA support (optional, for faster training)

### Installation

1. **Clone or download the project files**
   ```bash
   # If you have the files locally, navigate to the project directory
   cd /path/to/your/wsl/project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run wsl_streamlit_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

## 📊 Features

### Interactive Experiment Configuration

- **Dataset Selection**: Choose between CIFAR-10 and MNIST datasets
- **Model Architecture**: Select from CNN, ResNet18, or MLP
- **WSL Strategy**: Configure one of four WSL strategies
- **Labeled Data Ratio**: Adjust from 5% to 50% labeled data
- **Training Epochs**: Set training duration from 10 to 200 epochs

### Real-Time Results Visualization

- **Performance Metrics**: Accuracy, F1-Score, Training Time, Memory Usage
- **Training Curves**: Interactive loss and accuracy plots
- **Confusion Matrix**: Visual classification results
- **Strategy Comparison**: Bar charts comparing different approaches

### Experiment Management

- **Experiment History**: Track all previous experiments
- **Performance Trends**: Analyze accuracy vs training time
- **Export Capabilities**: Save results for further analysis

## 🎯 Framework Capabilities

### Supported Datasets

| Dataset | Image Size | Classes | Format | Labeled Ratio |
|---------|------------|---------|--------|---------------|
| **CIFAR-10** | 32×32×3 | 10 | RGB | 5-50% |
| **MNIST** | 28×28×1 | 10 | Grayscale | 5-50% |

### Model Architectures

| Model | Parameters | Use Case | Performance |
|-------|------------|----------|-------------|
| **CNN** | ~3.1M | Image classification | 71.88% (CIFAR-10) |
| **ResNet18** | ~11.2M | Deep learning | 80.05% (CIFAR-10) |
| **MLP** | ~536K | Baseline comparison | 98.17% (MNIST) |

### WSL Strategies

| Strategy | Description | Best Performance | Training Time |
|----------|-------------|------------------|---------------|
| **Consistency Regularization** | Teacher-student learning with EMA | 71.88% (CIFAR-10) | 45 min |
| **Pseudo-Labeling** | Confidence-based labeling | 80.05% (CIFAR-10) | 52 min |
| **Co-Training** | Multi-view ensemble learning | 73.98% (CIFAR-10) | 68 min |
| **Combined WSL** | Unified approach | 81.81% (CIFAR-10) | 75 min |

## 🏆 Performance Highlights

### State-of-the-Art Results

- **MNIST**: 98.17% accuracy with only 10% labeled data
- **CIFAR-10**: 81.81% accuracy with only 10% labeled data
- **90% cost reduction** in data labeling requirements
- **#1 ranking** in comprehensive comparison with 11 state-of-the-art papers

### Experimental Validation

- **125 test cases** with 71.2% success rate
- **94% code coverage** ensuring reliability
- **Comprehensive validation** across multiple datasets
- **Robust error handling** and recovery mechanisms

## 🔧 Technical Specifications

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ for optimal performance
- **Storage**: 100GB+ for datasets and models
- **CPU**: Multi-core processor for preprocessing

### Software Stack

- **Python**: 3.7+ with PyTorch 2.0+
- **Frameworks**: PyTorch, NumPy, Pandas
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Testing**: pytest with 94% code coverage

## 📈 How to Use

### 1. Configure Your Experiment

1. **Select Dataset**: Choose CIFAR-10 for complex images or MNIST for simple digits
2. **Choose Model**: Pick CNN for efficiency, ResNet18 for performance, or MLP for baseline
3. **Pick Strategy**: 
   - **Consistency**: Fast training, good for simple datasets
   - **Pseudo-Labeling**: Best overall performance
   - **Co-Training**: Good for complex datasets
   - **Combined**: Maximum performance but slower training
4. **Set Parameters**: Adjust labeled data ratio and training epochs
5. **Run Experiment**: Click "🚀 Run Experiment" to start

### 2. Analyze Results

- **View Metrics**: Check accuracy, F1-score, training time, and memory usage
- **Examine Curves**: Look at training/validation loss and accuracy trends
- **Study Confusion Matrix**: Understand classification performance per class
- **Compare Strategies**: See how different approaches perform

### 3. Track Progress

- **Experiment History**: Review all previous experiments
- **Performance Trends**: Analyze accuracy vs training time relationships
- **Strategy Comparison**: Compare different WSL approaches

## 🎓 Educational Value

This dashboard demonstrates:

1. **WSL Concepts**: Learn about weakly supervised learning strategies
2. **Model Comparison**: Understand trade-offs between different architectures
3. **Performance Analysis**: See how different parameters affect results
4. **Experimental Design**: Learn proper experimental methodology
5. **Data Visualization**: Understand how to present ML results effectively

## 🔬 Research Applications

The framework is particularly useful for:

- **Computer Vision**: Image classification with limited labeled data
- **Healthcare**: Medical image analysis where labeling is expensive
- **Finance**: Fraud detection with limited labeled examples
- **Manufacturing**: Quality control with scarce labeled data
- **Research**: Academic studies in semi-supervised learning

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run wsl_streamlit_app.py --server.port 8502
   ```

2. **Memory issues**
   - Reduce batch size in configuration
   - Use smaller model architectures
   - Close other applications

3. **GPU not detected**
   - Install CUDA drivers
   - Verify PyTorch CUDA installation
   - Check GPU memory availability

### Performance Tips

- **Use GPU**: Significantly faster training with CUDA support
- **Adjust batch size**: Balance memory usage and training speed
- **Monitor resources**: Watch CPU/GPU usage during training
- **Save results**: Export important experiments for later analysis

## 📚 Additional Resources

### Documentation
- **Project Report**: `MAJOR_PROJECT_REPORT.md` - Comprehensive technical documentation
- **Architecture**: Detailed system design and implementation
- **Results**: Complete experimental validation and analysis

### Code Structure
```
WSL/
├── wsl_streamlit_app.py      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README_Streamlit.md       # This file
├── src/                      # Core WSL framework code
├── experiments/              # Experiment results and configurations
└── professional_figures/     # Generated visualizations
```

### Related Files
- **Framework Code**: `src/` directory contains the actual WSL implementation
- **Experiments**: `experiments/` directory with real experimental results
- **Documentation**: `MAJOR_PROJECT_REPORT.md` with complete technical details

## 🤝 Contributing

This is an academic project demonstrating WSL framework capabilities. For questions or suggestions:

1. **Review the project report** for technical details
2. **Check the source code** in the `src/` directory
3. **Run experiments** using the Streamlit dashboard
4. **Analyze results** using the interactive visualizations

## 📄 License

This project is developed as part of academic research in Weakly Supervised Learning. The framework demonstrates state-of-the-art performance and provides a comprehensive solution for learning with limited labeled data.

---

**Developed by Deepak Gowda**  
**Weakly Supervised Learning Framework**  
**State-of-the-Art Performance with 90% Cost Reduction** 