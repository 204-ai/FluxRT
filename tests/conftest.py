import os
import sys

# Make scripts/ (batch_render, batch_routes) importable without installing them.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
