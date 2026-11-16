"""Asset checker regression tests using only the Python standard library."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT=Path(__file__).parents[1]

class AssetTests(unittest.TestCase):
    def test_manifest_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=b'asset sample'
            target=root/'sample.bin';target.write_bytes(data)
            manifest={'files':[{'file':target.name,'bytes':len(data),
                                'crc32':f'{zlib.crc32(data)&0xffffffff:08x}'}]}
            (root/'asset_manifest.json').write_text(json.dumps(manifest))
            def run():
                return subprocess.run([sys.executable,str(ROOT/'scripts/verify_assets.py'),directory],
                                      capture_output=True,text=True)
            result=run()
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('byte length and CRC32',result.stdout)
            for broken in (b'ASSET SAMPLE',b'truncated'):
                with self.subTest(content=broken):
                    target.write_bytes(broken)
                    result=run()
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(target.name,result.stderr)

if __name__=='__main__':unittest.main()
