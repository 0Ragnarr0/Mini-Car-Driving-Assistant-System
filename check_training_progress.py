#!/usr/bin/env python3
"""
Check RL training progress without running simulation.
Shows checkpoint files, loss trends, and episode metrics.
"""

import os
import sys
import numpy as np
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent

def check_checkpoints():
    """List saved RL checkpoints."""
    models_dir = PROJECT_ROOT / "trained_models"
    
    print("\n" + "="*70)
    print("RL TRAINING PROGRESS CHECK")
    print("="*70 + "\n")
    
    # Find all actor checkpoints
    actor_files = sorted(models_dir.glob("ddpg_autonomous_driving_ep_*_actor.pth"))
    
    if not actor_files:
        print("❌ NO CHECKPOINTS FOUND")
        print("   This means either:")
        print("   1. Training hasn't started yet (NAWNAW_MODE != 'rl')")
        print("   2. Training is still in warmup phase (first 2000 steps)")
        print("   3. No episodes have completed yet")
        print("\n   Expected checkpoints at: trained_models/ddpg_autonomous_driving_ep_*.pth")
        return False
    
    print(f"✓ Found {len(actor_files)} checkpoints:\n")
    
    for actor_file in actor_files[-5:]:  # Show last 5
        critic_file = actor_file.with_name(actor_file.name.replace("_actor", "_critic"))
        ep_num = re.search(r'ep_(\d+)', actor_file.name)
        ep_num = ep_num.group(1) if ep_num else "?"
        
        actor_size = actor_file.stat().st_size / (1024*1024)
        critic_size = critic_file.stat().st_size / (1024*1024) if critic_file.exists() else 0
        mod_time = actor_file.stat().st_mtime
        
        print(f"  Episode {ep_num:3s} | actor={actor_size:.1f}MB, critic={critic_size:.1f}MB | "
              f"modified: {Path(actor_file).stat().st_mtime}")
    
    print("\n")
    return True


def check_training_logs():
    """Look for training logs if they exist."""
    logs_dir = PROJECT_ROOT / "logs"
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.txt"))
        if log_files:
            print(f"✓ Found {len(log_files)} log files in logs/")
            print("   (Check these for loss values and training metrics)")
            return True
    
    return False


def check_data_collection():
    """Check if training data was collected."""
    data_dir = PROJECT_ROOT / "training_data"
    
    if data_dir.exists():
        batch_files = list(data_dir.glob("batch_*.npz"))
        if batch_files:
            total_size = sum(f.stat().st_size for f in batch_files) / (1024*1024)
            print(f"\n✓ Training data collected:")
            print(f"  {len(batch_files)} batch files, {total_size:.1f}MB total")
            return True
    
    return False


def get_training_status():
    """Determine current training status."""
    models_dir = PROJECT_ROOT / "trained_models"
    actor_files = list(models_dir.glob("ddpg_autonomous_driving_ep_*_actor.pth"))
    
    if actor_files:
        latest = max(actor_files, key=lambda f: f.stat().st_mtime)
        ep_num = re.search(r'ep_(\d+)', latest.name)
        episodes = int(ep_num.group(1)) if ep_num else 0
        
        print(f"\n✓ TRAINING IS ACTIVE")
        print(f"  Latest checkpoint: episode {episodes}")
        print(f"  Last saved: {latest.stat().st_mtime}")
        
        # Estimate progress
        if episodes >= 100:
            print(f"  Status: ✓✓✓ Good progress ({episodes} episodes)")
        elif episodes >= 20:
            print(f"  Status: ✓✓ Moderate progress ({episodes} episodes)")
        else:
            print(f"  Status: ✓ Early stage ({episodes} episodes)")
        
        return True
    else:
        print(f"\n⏳ NOT TRAINING YET")
        print(f"   Start training with: $env:NAWNAW_MODE = 'rl'")
        return False


if __name__ == "__main__":
    has_checkpoints = check_checkpoints()
    check_data_collection()
    check_training_logs()
    get_training_status()
    
    print("\n" + "="*70)
    print("HOW TO VERIFY TRAINING IS WORKING:")
    print("="*70)
    print("""
1. Run simulation with: $env:NAWNAW_MODE = "rl"
2. Watch console output for:
   - ✓LEARNING indicator (after 2000 warmup steps)
   - actor_loss and critic_loss values (should decrease over time)
   - Checkpoints saved every 10 episodes
3. Run this script again: python check_training_progress.py
4. Verify checkpoints are created/updated in trained_models/
""")
