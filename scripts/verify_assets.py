
"""Byte length + CRC32 integrity. Not a cryptographic authenticity guarantee."""
import argparse,json,zlib,hashlib
from pathlib import Path
def crc(path):
    v=0
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):v=zlib.crc32(chunk,v)
    return f'{v&0xffffffff:08x}'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
    m=json.loads((root/'asset_manifest.json').read_text())
    for r in m['files']:
        q=root/r['file']
        if q.stat().st_size!=r['bytes'] or crc(q)!=r['crc32'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(r['file'])
    print(f"PASS: {len(m['files'])} files; byte length, CRC32 and SHA256")
