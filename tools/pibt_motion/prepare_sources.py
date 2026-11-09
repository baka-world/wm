"""Download the four pre-pinned hospital source files with hash verification."""
import argparse
from pathlib import Path
from experiments.pibt_motion.hospital_reproduction import NAMES
from lab_transport.training_data import download_sources

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('refusing to overwrite source files')
a.output.mkdir(parents=True,exist_ok=True)
for name,raw in download_sources().items():(a.output/NAMES[name]).write_bytes(raw)
print('downloaded and verified four pinned public sources')
