#!/usr/bin/env python3
from pathlib import Path
import shutil, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'; DATA=ROOT/'results'/'frozen_summaries'

def main():
    target=PAPER/'experiment_data'
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(DATA,target)
    root_inputs = ['r1_cumulative_cost_curves.csv','s1_holdout_global_comparison_summary.csv','b3c_balanced_interference_summary.csv']
    copied = []
    for name in root_inputs:
        src = DATA / name
        if src.exists():
            dst = PAPER / name
            shutil.copy2(src, dst); copied.append(dst)
    try:
        subprocess.run([sys.executable,str(PAPER/'make_paper_assets.py')],cwd=PAPER,check=True)
        subprocess.run([sys.executable,str(PAPER/'make_cleaned_figures.py')],cwd=PAPER,check=True)
    finally:
        shutil.rmtree(target,ignore_errors=True)
        for p in copied:
            p.unlink(missing_ok=True)
    print('paper assets regenerated')
if __name__=='__main__': main()
