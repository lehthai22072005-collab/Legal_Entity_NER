"""Validate the current 369-document dataset (compatibility entry point)."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_all_data import main
if __name__=='__main__':main()
