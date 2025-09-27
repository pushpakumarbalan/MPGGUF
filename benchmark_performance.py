#!/usr/bin/env python3
"""
Performance benchmarking script for MPGGUF vs standard GGUF files.
Compares inference speed, memory usage, and file sizes across different quantization strategies.
"""

import time
import psutil
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import json

class MPGGUFBenchmark:
    """Benchmark suite for MPGGUF performance analysis"""
    
    def __init__(self, models_dir: str = "/Users/pushpakumar/Projects/VScodeProjects/models/qwen2.5-7b"):
        self.models_dir = Path(models_dir)
        self.llama_cpp_path = Path("/Users/pushpakumar/Projects/VScodeProjects/llama.cpp/build/bin")
        
    def analyze_file_sizes(self) -> Dict[str, Any]:
        """Analyze file sizes and compression ratios"""
        models = {
            'F16 (Baseline)': self.models_dir / 'qwen2.5-7b.F16.gguf',
            'Q8_0 (High Precision)': self.models_dir / 'qwen2.5-7b.Q8_0.gguf', 
            'Q4_0 (Medium Precision)': self.models_dir / 'qwen2.5-7b.Q4_0.gguf',
            'Q2_K (Low Precision)': self.models_dir / 'qwen2.5-7b.Q2_K.gguf',
            'MPGGUF (Mixed)': self.models_dir / 'qwen2.5-7b-mixed.mpgguf'
        }
        
        results = {}
        baseline_size = None
        
        for name, path in models.items():
            if path.exists():
                size_bytes = path.stat().st_size
                size_gb = size_bytes / (1024**3)
                
                if 'Baseline' in name:
                    baseline_size = size_bytes
                    compression_ratio = 1.0
                else:
                    compression_ratio = size_bytes / baseline_size if baseline_size else 1.0
                
                results[name] = {
                    'size_bytes': size_bytes,
                    'size_gb': round(size_gb, 2),
                    'compression_ratio': round(compression_ratio, 3),
                    'exists': True
                }
            else:
                results[name] = {'exists': False}
        
        return results
    
    def estimate_memory_usage(self, file_path: Path) -> Dict[str, float]:
        """Estimate memory requirements for model loading"""
        if not file_path.exists():
            return {'model_size_gb': 0, 'estimated_ram_gb': 0}
        
        model_size_gb = file_path.stat().st_size / (1024**3)
        # Rough estimate: model size + overhead for inference
        estimated_ram_gb = model_size_gb * 1.2  # 20% overhead estimate
        
        return {
            'model_size_gb': round(model_size_gb, 2),
            'estimated_ram_gb': round(estimated_ram_gb, 2)
        }
    
    def benchmark_loading_time(self, file_path: Path) -> Optional[float]:
        """Benchmark model loading time (if llama.cpp is available)"""
        if not file_path.exists():
            return None
        
        llama_cli = self.llama_cpp_path / "llama-cli"
        if not llama_cli.exists():
            return None
        
        try:
            start_time = time.time()
            # Run a minimal inference to test loading
            result = subprocess.run([
                str(llama_cli), 
                "-m", str(file_path),
                "-p", "Hello",
                "-n", "1",
                "--no-display-prompt"
            ], capture_output=True, text=True, timeout=60)
            
            loading_time = time.time() - start_time
            
            if result.returncode == 0:
                return round(loading_time, 2)
            else:
                return None
                
        except (subprocess.TimeoutExpired, Exception):
            return None
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🔥 Starting MPGGUF Performance Benchmark")
        print("=" * 60)
        
        # File size analysis
        print("📊 Analyzing file sizes...")
        size_results = self.analyze_file_sizes()
        
        # Memory analysis
        print("🧠 Analyzing memory requirements...")
        memory_results = {}
        for name, data in size_results.items():
            if data.get('exists', False):
                model_path = None
                for path in self.models_dir.glob("*.gguf"):
                    if any(key in str(path) for key in name.split()):
                        model_path = path
                        break
                if not model_path and 'MPGGUF' in name:
                    model_path = self.models_dir / 'qwen2.5-7b-mixed.mpgguf'
                
                if model_path and model_path.exists():
                    memory_results[name] = self.estimate_memory_usage(model_path)
        
        # System info
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
        
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': system_info,
            'file_sizes': size_results,
            'memory_estimates': memory_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way"""
        print("\n" + "=" * 60)
        print("📈 MPGGUF PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        # System Info
        sys_info = results['system_info']
        print(f"\n💻 System Configuration:")
        print(f"  CPU Cores: {sys_info['cpu_count']}")
        print(f"  Total RAM: {sys_info['memory_gb']} GB")
        print(f"  Available RAM: {sys_info['available_memory_gb']} GB")
        
        # File Sizes
        print(f"\n📁 File Size Analysis:")
        for name, data in results['file_sizes'].items():
            if data.get('exists', False):
                status = "✅"
                size_info = f"{data['size_gb']} GB"
                ratio_info = f"({data['compression_ratio']:.1%} of baseline)"
                print(f"  {status} {name:20} {size_info:8} {ratio_info}")
            else:
                print(f"  ❌ {name:20} {'Not found':8}")
        
        # Memory Estimates  
        if results.get('memory_estimates'):
            print(f"\n🧠 Memory Requirements:")
            for name, data in results['memory_estimates'].items():
                print(f"  {name:20} Model: {data['model_size_gb']} GB, Est. RAM: {data['estimated_ram_gb']} GB")
        
        # Efficiency Summary
        file_sizes = results['file_sizes']
        if 'MPGGUF (Mixed)' in file_sizes and file_sizes['MPGGUF (Mixed)'].get('exists'):
            mpgguf_ratio = file_sizes['MPGGUF (Mixed)']['compression_ratio']
            print(f"\n🎯 MPGGUF Efficiency:")
            print(f"  Size vs Baseline: {mpgguf_ratio:.1%}")
            print(f"  Space Saved: {(1-mpgguf_ratio):.1%}")
        
        print(f"\n⏰ Benchmark completed at: {results['timestamp']}")


def main():
    """Main benchmark execution"""
    benchmark = MPGGUFBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_results(results)
        
        # Save results to file
        output_file = Path("benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())