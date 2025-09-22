import os
import sys
import signal
import multiprocessing
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Set environment variables to prevent multiprocessing issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'
os.environ['LAPACK_NUM_THREADS'] = '1'
os.environ['ATLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

# Set multiprocessing start method to avoid resource leaks
import multiprocessing
if multiprocessing.get_start_method() != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import torch after setting environment variables
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Set torch multiprocessing settings
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)

# Import local modules
from src.data.preprocessing import get_cifar10_dataloaders, get_mnist_dataloaders
from src.data.clothing1m import get_clothing1m_dataloaders
from src.models.baseline import SimpleCNN, ResNet, MLP
from src.models.noise_robust import RobustCNN, RobustResNet
from src.training.train import Trainer

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    print(f"\nReceived signal {signum}. Cleaning up and exiting...")
    
    # Force cleanup of any remaining processes
    try:
        # Clean up multiprocessing resources
        multiprocessing.current_process()._cleanup()
        
        # Kill any remaining child processes
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.terminate()
                child.wait(timeout=5)
            except:
                try:
                    child.kill()
                except:
                    pass
    except:
        pass
    
    # Force cleanup of torch resources
    try:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except:
        pass
    
    # Force cleanup of any remaining semaphores
    try:
        # Clean up multiprocessing semaphores
        import multiprocessing.resource_tracker
        multiprocessing.resource_tracker._CLEANUP_CALLS.clear()
    except:
        pass
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments with noisy labels')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'mnist', 'clothing1m'],
                      help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--noise_type', type=str, default='random',
                      choices=['random', 'instance_dependent'],
                      help='Type of label noise')
    parser.add_argument('--noise_rate', type=float, default=0.1,
                      help='Rate of label noise')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to use (for quick testing)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='robust_cnn',
                      choices=['simple_cnn', 'resnet', 'robust_cnn', 'robust_resnet', 'mlp', 'robust_mlp'],
                      help='Type of model to use')
    parser.add_argument('--loss_type', type=str, default='gce',
                      choices=['ce', 'gce', 'sce'],
                      help='Type of loss function')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=10,
                      help='Early stopping patience')
    
    return parser.parse_args()

def get_dataloaders(args: argparse.Namespace) -> Dict[str, Any]:
    """Get dataloaders based on dataset choice"""
    if args.dataset == 'cifar10':
        return get_cifar10_dataloaders(
            batch_size=args.batch_size,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate
        )
    elif args.dataset == 'mnist':
        return get_mnist_dataloaders(
            batch_size=args.batch_size,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate
        )
    elif args.dataset == 'clothing1m':
        return get_clothing1m_dataloaders(
            root_dir='./data/clothing1m',
            batch_size=args.batch_size,
            noise_rate=args.noise_rate,
            num_samples=args.num_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def main():
    """Main experiment function"""
    parser = argparse.ArgumentParser(description='Run noise-robust training experiment')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist', 'clothing1m'])
    parser.add_argument('--model_type', type=str, default='robust_cnn', 
                       choices=['simple_cnn', 'resnet', 'mlp', 'robust_cnn', 'robust_resnet', 'robust_mlp'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='gce', choices=['gce', 'sce', 'forward'])
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.dataset}_{args.model_type}_{args.loss_type}_noise{args.noise_rate}_{timestamp}"
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load data
        if args.dataset == 'cifar10':
            dataloaders = get_cifar10_dataloaders(
                batch_size=args.batch_size, noise_rate=args.noise_rate
            )
            train_loader = dataloaders['train']
            test_loader = dataloaders['test']
            
            # Create validation split from training data
            train_dataset = train_loader.dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
            
        elif args.dataset == 'mnist':
            dataloaders = get_mnist_dataloaders(
                batch_size=args.batch_size, noise_rate=args.noise_rate
            )
            train_loader = dataloaders['train']
            test_loader = dataloaders['test']
            
            # Create validation split from training data
            train_dataset = train_loader.dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
            
        elif args.dataset == 'clothing1m':
            dataloaders = get_clothing1m_dataloaders(
                batch_size=args.batch_size
            )
            train_loader = dataloaders['train']
            test_loader = dataloaders['test']
            
            # Create validation split from training data
            train_dataset = train_loader.dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        
        # Create model
        if args.model_type == 'simple_cnn':
            model = SimpleCNN(num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10)
        elif args.model_type == 'resnet':
            model = ResNet(num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10)
        elif args.model_type == 'mlp':
            model = MLP(input_size=784 if args.dataset == 'mnist' else 3072, 
                       num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10)
        elif args.model_type == 'robust_mlp':
            # Create MLP with robust loss capabilities
            model = MLP(input_size=784 if args.dataset == 'mnist' else 3072, 
                       num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10)
            # Add robust loss attributes to the model
            model.loss_type = args.loss_type
            model.current_epoch = 0
            model.q = 0.7  # Default q value for GCE
            model.alpha = 0.1  # Default alpha for SCE
            model.beta = 1.0  # Default beta for SCE
            
            # Import robust loss functions
            from src.models.noise_robust import GCE, SCE
            
            # Add loss functions to the model
            model.ce_loss = nn.CrossEntropyLoss()
            model.gce_loss = GCE(q=model.q)
            model.sce_loss = SCE(alpha=model.alpha, beta=model.beta)
            
            # Add compute_loss method to the model
            def compute_loss(outputs, targets, reduction='mean'):
                """Compute noise-robust loss with hybrid training"""
                # Use standard CE loss for first 5 epochs to stabilize training
                if model.current_epoch < 5:
                    return model.ce_loss(outputs, targets)
                
                # After 5 epochs, use the specified robust loss
                if model.loss_type == 'gce':
                    return model.gce_loss(outputs, targets)
                elif model.loss_type == 'sce':
                    return model.sce_loss(outputs, targets)
                else:
                    # Fallback to standard CE
                    return model.ce_loss(outputs, targets)
            
            def set_epoch(epoch):
                """Set current epoch for hybrid training"""
                model.current_epoch = epoch
            
            # Attach methods to the model
            model.compute_loss = compute_loss
            model.set_epoch = set_epoch
        elif args.model_type == 'robust_cnn':
            model = RobustCNN(num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10,
                             loss_type=args.loss_type)
        elif args.model_type == 'robust_resnet':
            model = RobustResNet(num_classes=10 if args.dataset in ['cifar10', 'clothing1m'] else 10,
                                loss_type=args.loss_type)
        
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            save_dir=experiment_dir
        )
        
        # Train model
        print(f"Starting training for {args.epochs} epochs...")
        trainer.train(epochs=args.epochs)
        
        # Test model
        print("Testing model...")
        trainer.evaluate()
        
        # Get test results from saved file
        test_results_path = experiment_dir / 'test_results.json'
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                test_metrics = json.load(f)
        else:
            test_metrics = {'test_accuracy': 0.0, 'test_loss': 0.0}
        
        # Save results
        results = {
            'dataset': args.dataset,
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'noise_rate': args.noise_rate,
            'loss_type': args.loss_type,
            'learning_rate': args.lr,
            'test_metrics': test_metrics,
            'timestamp': timestamp
        }
        
        with open(experiment_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        raise
    finally:
        # Final cleanup
        try:
            # Clean up torch resources
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Clean up multiprocessing resources
            multiprocessing.current_process()._cleanup()
            
            # Clean up dataloaders
            if 'train_loader' in locals():
                del train_loader
            if 'val_loader' in locals():
                del val_loader
            if 'test_loader' in locals():
                del test_loader
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except:
            pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user. Cleaning up...")
        # Clean up multiprocessing resources
        try:
            multiprocessing.current_process()._cleanup()
            import gc
            gc.collect()
        except:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"\nError during experiment: {e}")
        print("Cleaning up and exiting...")
        # Clean up multiprocessing resources
        try:
            multiprocessing.current_process()._cleanup()
            import gc
            gc.collect()
        except:
            pass
        sys.exit(1)
    finally:
        # Final cleanup
        try:
            # Clean up multiprocessing resources
            multiprocessing.current_process()._cleanup()
            
            # Clean up any remaining semaphores
            import multiprocessing.resource_tracker
            multiprocessing.resource_tracker._CLEANUP_CALLS.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except:
            pass