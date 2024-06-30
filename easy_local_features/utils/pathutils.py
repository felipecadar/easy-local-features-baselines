from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
CACHE_BASE = Path(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'easy_local_features')
