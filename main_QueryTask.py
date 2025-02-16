import os
import sys

# Ensure script runs from root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.get_tles.main_LEOQuery import main  # Adjust import as needed

if __name__ == "__main__":
    main()