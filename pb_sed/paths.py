import os
from pathlib import Path

pb_sed_root = Path(__file__).parent.parent
database_jsons_dir = Path(os.getenv('DATABASE_JSONS_DIR', str(pb_sed_root / 'jsons'))).expanduser()
storage_root = Path(os.getenv('STORAGE_ROOT', str(pb_sed_root / 'exp'))).expanduser()
