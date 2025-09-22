College
RV COLLEGE OF ENGINEERING®
Department
COMPUTER SCIENCE AND ENGINEERING
PROGRAM: M.Tech in
CSE
MAJOR PROJECT
Course Code
22MCE41P
Student Name
Deepak Ishwar Gouda
USN
1RV23SCS03
Project Title
Weakly Supervised Learning Framework Using Deep Learning Techniques
Undertaken at
RVCE
Internal Guide
 Dr. Shanta Rangaswamy
Prof. & HOD, Dept. of CSE, RVCE

INTRODUCTION:
Modern machine learning models typically require large amounts of high-quality labeled data for training. However, in many real-world applications such as medical imaging, autonomous vehicles, and industrial inspection, obtaining clean and abundant labeled data is challenging and expensive. This project addresses these challenges by developing a unified Weakly Supervised Learning (WSL) framework that can effectively train deep learning models with limited labeled data while maintaining strong generalization capabilities. The research focuses on WSL approaches including consistency regularization, pseudo-labeling, and co-training that can learn from both labeled and unlabeled data. By investigating various WSL paradigms and implementing multiple deep learning architectures including CNN, ResNet, and MLP, this project aims to develop practical solutions for training machine learning models in resource-constrained environments. 

OBJECTIVES:

1. To develop a unified WSL framework that combines multiple strategies for optimal performance with limited labeled data
2. To implement comprehensive support for multiple deep learning architectures including CNN, ResNet, and MLP models
3. To achieve state-of-the-art performance on benchmark datasets while maintaining computational efficiency and scalability
4. To integrate robust loss functions including GCE, SCE, and Forward Correction to handle noisy and inconsistent labels
5. To establish comprehensive evaluation metrics and benchmarking procedures for standardized assessment of WSL performance

METHODOLOGY:

1. Dataset Selection and Preparation
    Standard benchmarks: CIFAR-10, MNIST datasets
    Data preprocessing with augmentation techniques and normalization
    Train/validation/test splits with limited labeled data scenarios
    Robust data loading and batching for efficient training

2. WSL Strategy Implementation
    Consistency regularization: Enforcing consistency across different augmentations
    Pseudo-labeling: Using high-confidence predictions as labels for unlabeled data
    Co-training: Training multiple models on different views of data
    Combined approach: Integrating multiple WSL strategies for synergistic effects

3. Model Architecture Integration
    Convolutional Neural Networks (CNN): Feature extraction for image data
    ResNet: Residual connections for deeper networks
    Multi-Layer Perceptrons (MLP): Universal function approximation
    Robust loss functions: GCE, SCE, and Forward Correction for noise handling

4. Framework Development and Evaluation
    Unified framework implementation with modular design
    Comprehensive testing with 94% code coverage and 140 test cases
    Performance evaluation using standardized metrics
    Production-ready implementation with comprehensive documentation

SOFTWARE REQUIREMENTS:
 Python 3.8+
 PyTorch 1.9+
 Scikit-learn
 NumPy, Pandas
 Matplotlib, Seaborn
 Jupyter Notebook
 Pytest (for testing)

HARDWARE REQUIREMENTS:
 GPU-enabled system (NVIDIA GPU with CUDA support)
 Minimum 8GB RAM (16GB recommended)
 100GB SSD storage
 Multicore processor

INNOVATION / CONTRIBUTION TO THE FIELD:
1. Technical Innovation
    Novel unified framework combining multiple WSL strategies
    Implementation of robust loss functions for noisy label handling
    Comprehensive evaluation system with standardized metrics
    Production-ready framework with 94% code coverage

2. Practical Impact
    Open-source implementation of WSL techniques
    Significant performance improvements (98.26% accuracy on MNIST)
    90% reduction in labeling requirements while maintaining performance
    Comprehensive documentation and user guides for practitioners

3. Research Contribution
    Empirical analysis of WSL methods across multiple datasets
    Benchmark results for limited labeled data scenarios
    Standardized evaluation procedures for WSL research
    Reproducible implementation for the research community



     Internal Guide                                   Signature of CSE Dean Cluster                            Signature of the HOD
Dr.Shanta Rangaswamy                        Dr.Ramakanth Kumar P                                 Dr.Shanta Rangaswamy
  Professor and HOD	                  Professor and CSE Dean Cluster                               Professor and HOD
  Dept. of CSE, RVCE                                    Dept. of CSE, RVCE                                          Dept. of CSE, RVCE        
