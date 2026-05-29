"""Import this first. Sets up module paths and the working directory so all relative
paths (data/, checkpoints/, body_models/) resolve regardless of where you invoke from."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
for p in (ROOT, os.path.join(ROOT, 'motionlib')):
    if p not in sys.path:
        sys.path.insert(0, p)
