"""Copy final analysis files and verify their remote hashes."""
import base64
import json
from pathlib import Path
import shlex
from remote import remote
from common import REMOTE,TRAIN_PYTHON,digest,write_json

ROOT=Path(__file__).resolve().parent
PROGRAM=r"""
import base64,hashlib,json
from pathlib import Path
root=Path(ROOT);folder=root/'analysis';files={};hashes={}
for path in folder.iterdir():
 if path.is_file() and path.suffix in {'.json','.csv','.md','.png','.pdf'}:
  data=path.read_bytes();files[path.name]=base64.b64encode(data).decode();hashes[path.name]=hashlib.sha256(data).hexdigest()
if 'complete.json' in files:
 proof=json.loads((folder/'complete.json').read_text())
 assert proof['status']=='completed'
 assert proof['source_bundle_sha256']==EXPECTED
 for name,expected in proof['files_sha256'].items():assert hashes[name]==expected,'Final analysis artifact changed: '+name
print(json.dumps({'files':files,'files_sha256':hashes,'complete':'complete.json' in files}))
"""

def main():
    capture=json.loads(remote(shlex.quote(TRAIN_PYTHON)+' -','ROOT='+repr(REMOTE)+'\nEXPECTED='+repr(digest(ROOT/'ANALYSIS_JOB_BUNDLE.json'))+'\n'+PROGRAM,timeout=120))
    files=capture.pop('files');out=ROOT/'analysis';out.mkdir(exist_ok=True)
    final=(out/'complete.json').exists()
    for name in sorted(files,key=lambda name:name=="complete.json"):
        encoded=files[name]
        path=(out/name).resolve()
        if not path.is_relative_to(out.resolve()):raise ValueError('Unsafe analysis output path')
        data=base64.b64decode(encoded,validate=True)
        import hashlib
        if hashlib.sha256(data).hexdigest()!=capture['files_sha256'][name]:raise ValueError('Copied analysis hash differs')
        if final and path.exists() and path.read_bytes()!=data:raise ValueError('A saved final analysis artifact changed')
        temporary=path.with_name(path.name+".sync.tmp")
        temporary.write_bytes(data);temporary.replace(path)
    write_json(ROOT/'ANALYSIS_SYNC_RECEIPT.json',capture)
    print(json.dumps({'analysis_complete':capture['complete'],'files_copied':len(files)}))

if __name__=='__main__':main()
