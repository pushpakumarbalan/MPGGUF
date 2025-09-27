#!/usr/bin/env python3
"""
Download target models for MPGGUF testing

Downloads the specified models from HuggingFace Hub for conversion to GGUF format.
For testing, we'll start with a smaller model before moving to the large ones.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict

try:
    from huggingface_hub import snapshot_download, login
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    print(f"Error: Missing required packages. Please install requirements.txt")
    print(f"Details: {e}")
    exit(1)


class ModelDownloader:
    """Download and prepare models for GGUF conversion"""
    
    # Target models from the task specification
    TARGET_MODELS = {
        'qwen3-235b-a22b': {
            'repo_id': 'Qwen/Qwen3-235B-A22B',
            'description': 'Qwen3 235B parameters (A22B active)',
            'size_gb': '~500GB',
            'priority': 2  # Large model - test after small one
        },
        'glm-4.5': {
            'repo_id': 'zai-org/GLM-4.5', 
            'description': 'GLM-4.5 model',
            'size_gb': '~50GB',
            'priority': 2  # Large model
        },
        'llama-3.2-1b': {
            'repo_id': 'meta-llama/Llama-3.2-1B-Instruct',
            'description': 'Llama 3.2 1B (for testing)',
            'size_gb': '~2GB',
            'priority': 1  # Small model for testing first
        },
        'qwen2.5-7b': {
            'repo_id': 'Qwen/Qwen2.5-7B-Instruct',
            'description': 'Qwen2.5 7B (medium size for testing)',
            'size_gb': '~14GB', 
            'priority': 1  # Medium model for testing
        }
    }
    
    def __init__(self, models_dir: str = "../models", use_auth_token: bool = False):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.use_auth_token = use_auth_token
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def list_models(self) -> None:
        """List available models with details"""
        print("Available Models for Download:")
        print("=" * 60)
        
        for model_key, info in self.TARGET_MODELS.items():
            priority_str = "🔥 HIGH" if info['priority'] == 1 else "📋 NORMAL"
            print(f"Key: {model_key}")
            print(f"  Repository: {info['repo_id']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size_gb']}")
            print(f"  Priority: {priority_str}")
            
            # Check if already downloaded
            model_path = self.models_dir / model_key
            if model_path.exists():
                print(f"  Status: ✅ Already downloaded to {model_path}")
            else:
                print(f"  Status: ⬇️  Available for download")
            print()
    
    def download_model(self, model_key: str, force: bool = False) -> str:
        """
        Download a specific model
        
        Args:
            model_key: Key from TARGET_MODELS
            force: Re-download even if already exists
            
        Returns:
            Path to downloaded model directory
        """
        if model_key not in self.TARGET_MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Use --list to see available models.")
        
        model_info = self.TARGET_MODELS[model_key]
        repo_id = model_info['repo_id']
        
        # Check if model already exists
        model_path = self.models_dir / model_key
        if model_path.exists() and not force:
            self.logger.info(f"Model {model_key} already exists at {model_path}")
            return str(model_path)
        
        self.logger.info(f"Downloading {model_key} from {repo_id}...")
        self.logger.info(f"Expected size: {model_info['size_gb']}")
        
        try:
            # Login if using auth token
            if self.use_auth_token:
                self.logger.info("Attempting to login with HF token...")
                login()  # Uses HF_TOKEN env var or cached token
            
            # Download the model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # Copy files instead of symlinks
                resume_download=True,  # Resume partial downloads
                ignore_patterns=[
                    "*.bin",  # Skip older PyTorch format
                    "pytorch_model.bin",
                    "*.h5",   # Skip TensorFlow format
                    "*.tflite",
                    "*.ot",   # Skip ONNX format
                ]
            )
            
            self.logger.info(f"✅ Successfully downloaded {model_key} to {model_path}")
            
            # Verify essential files exist
            self._verify_model(model_path)
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_key}: {e}")
            
            # Check if it's an auth issue
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self.logger.error("Authentication failed. For gated models, you need:")
                self.logger.error("1. Accept model license on HuggingFace Hub")
                self.logger.error("2. Login: huggingface-cli login")
                self.logger.error("3. Or set HF_TOKEN environment variable")
                self.logger.error("4. Re-run with --auth flag")
            
            # Cleanup failed download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            raise
    
    def _verify_model(self, model_path: Path) -> None:
        """Verify that essential model files are present"""
        required_files = ['config.json']
        safetensors_files = list(model_path.glob("*.safetensors"))
        
        # Check for config
        config_path = model_path / 'config.json'
        if not config_path.exists():
            raise ValueError(f"Missing config.json in {model_path}")
        
        # Check for model weights
        if not safetensors_files:
            raise ValueError(f"No .safetensors files found in {model_path}")
        
        # Check for tokenizer files
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']
        missing_tokenizer = []
        for tf in tokenizer_files:
            if not (model_path / tf).exists():
                missing_tokenizer.append(tf)
        
        if missing_tokenizer:
            self.logger.warning(f"Missing tokenizer files: {missing_tokenizer}")
            self.logger.warning("This may cause issues during GGUF conversion")
        
        self.logger.info(f"Model verification passed: {len(safetensors_files)} weight files found")
    
    def download_testing_models(self) -> List[str]:
        """Download recommended models for testing (priority 1)"""
        self.logger.info("Downloading recommended models for testing...")
        
        testing_models = [k for k, v in self.TARGET_MODELS.items() if v['priority'] == 1]
        downloaded = []
        
        for model_key in testing_models:
            try:
                path = self.download_model(model_key)
                downloaded.append(path)
            except Exception as e:
                self.logger.error(f"Failed to download {model_key}: {e}")
        
        return downloaded
    
    def get_model_path(self, model_key: str) -> Path:
        """Get the local path for a model"""
        return self.models_dir / model_key


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download models for MPGGUF testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python download_models.py --list
  
  # Download recommended testing models
  python download_models.py --test
  
  # Download a specific model
  python download_models.py --model qwen2.5-7b
  
  # Download with authentication for gated models
  python download_models.py --model llama-3.2-1b --auth
  
  # Force re-download
  python download_models.py --model qwen2.5-7b --force
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--test', action='store_true', help='Download recommended testing models')
    parser.add_argument('--model', help='Download specific model by key')
    parser.add_argument('--auth', action='store_true', help='Use HuggingFace authentication')
    parser.add_argument('--force', action='store_true', help='Force re-download existing models')
    parser.add_argument('--models-dir', default='../models', help='Models directory (default: ../models)')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ModelDownloader(args.models_dir, args.auth)
    
    try:
        if args.list:
            downloader.list_models()
            
        elif args.test:
            print("Downloading recommended testing models...")
            downloaded = downloader.download_testing_models()
            print(f"\n✅ Downloaded {len(downloaded)} models for testing")
            for path in downloaded:
                print(f"  - {path}")
            
        elif args.model:
            print(f"Downloading model: {args.model}")
            path = downloader.download_model(args.model, args.force)
            print(f"✅ Model downloaded to: {path}")
            
        else:
            # Default: show list and ask user
            downloader.list_models()
            print("\nUse --test to download testing models, or --model <key> for a specific model")
            print("For gated models (like Llama), add --auth flag")
            
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()