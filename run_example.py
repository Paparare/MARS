#!/usr/bin/env python3
"""
Example script demonstrating how to run the benchmark framework.

This script shows how to:
1. Run a baseline benchmark
2. Generate prompt enhancements from failures
3. Re-run with enhanced prompts

Usage:
    python run_example.py
"""

import os
import subprocess
import sys
from pathlib import Path

def check_api_keys():
    """Check if required API keys are set."""
    openai_key = os.environ.get('OPENAI_API_KEY')
    together_key = os.environ.get('TOGETHER_API_KEY')
    
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return False
    
    if not together_key:
        print("⚠️  Warning: TOGETHER_API_KEY not set (needed for enhancement generation)")
        print("   Set it with: export TOGETHER_API_KEY='your-key'")
    
    return True

def check_data():
    """Check if GPQA data files exist."""
    data_dir = Path('./data')
    train_file = data_dir / 'gpqa_diamond_train.csv'
    test_file = data_dir / 'gpqa_diamond_test.csv'
    
    if not train_file.exists() and not test_file.exists():
        print("⚠️  Warning: No GPQA data files found in ./data/")
        print("   Please add gpqa_diamond_train.csv or gpqa_diamond_test.csv")
        return False
    
    return True

def run_baseline_benchmark():
    """Run a small baseline benchmark."""
    print("\n" + "="*60)
    print("Step 1: Running Baseline Benchmark (Zero-Shot CoT)")
    print("="*60)
    
    cmd = [
        sys.executable,
        'strategies/zero-shot-cot.py',
        '--data-file', './data/gpqa_diamond_train.csv',
        '--prompt-type', 'zero_shot_cot',
        '--start-index', '0',
        '--end-index', '10',  # Only run 10 questions for demo
        '--model', 'gpt-4o-mini',  # Use cheaper model for demo
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_enhancement_generation(results_dir):
    """Generate prompt enhancements from failures."""
    print("\n" + "="*60)
    print("Step 2: Generating Prompt Enhancements")
    print("="*60)
    
    cmd = [
        sys.executable,
        'strategies/zero-shot-cot-enhancement.py',
        '--input', results_dir,
        '--output-dir', './enhanced_prompts',
        '--enhancement-types', 'concise', 'specific', 'reasoning',
        '--max-questions', '5',  # Limit for demo
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    print("="*60)
    print("Prompt Enhancement Benchmark Framework - Example")
    print("="*60)
    
    # Check prerequisites
    if not check_api_keys():
        print("\nPlease set the required API keys and try again.")
        return
    
    if not check_data():
        print("\nPlease add GPQA data files and try again.")
        return
    
    print("\n✓ Prerequisites check passed!")
    
    # Run baseline benchmark
    print("\nThis example will:")
    print("  1. Run a baseline benchmark on 10 questions")
    print("  2. Generate prompt enhancements from failures")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    run_baseline_benchmark()
    
    # Find results directory
    from pathlib import Path
    results_dirs = sorted(Path('.').glob('gpqa_results_*'))
    if results_dirs:
        latest_results = str(results_dirs[-1])
        print(f"\n✓ Found results in: {latest_results}")
        
        print("\nGenerate enhancements? (y/n)")
        if input().lower() == 'y':
            run_enhancement_generation(latest_results)
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check ./results/ for benchmark outputs")
    print("  2. Check ./enhanced_prompts/ for generated enhancements")
    print("  3. Run full benchmark with: python strategies/zero-shot-cot.py --help")

if __name__ == '__main__':
    main()
