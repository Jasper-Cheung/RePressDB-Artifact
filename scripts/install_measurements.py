"""Install the checksum-verified event arrays from the measurement archive."""
from pathlib import Path
from zipfile import ZipFile
import hashlib,sys

ARCHIVE_SHA256='2fb8c9a13be63e6359dab59a8df26a792745d5bb19b17e07f222cfe6117b9925'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python scripts/install_measurements.py ARCHIVE.zip')
    archive=Path(sys.argv[1]);assert digest(archive)==ARCHIVE_SHA256,'Archive checksum mismatch'
    target=Path(__file__).resolve().parents[1]/'experiments/current';count=0
    with ZipFile(archive) as z:
        for entry in z.infolist():
            marker='/FGCS_revision/experiments/'
            if marker not in entry.filename or not entry.filename.endswith('.npz'):continue
            relative=Path(entry.filename.split(marker,1)[1]);assert not relative.is_absolute() and '..' not in relative.parts
            path=target/relative;data=z.read(entry)
            if path.exists():assert hashlib.sha256(data).hexdigest()==digest(path),str(relative)
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            count+=1
    print(f'Installed or verified {count} measurement arrays.')
if __name__=='__main__':main()
