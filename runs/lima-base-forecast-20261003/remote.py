"""Publish, run, and inspect the separate remote campaign."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
from common import REMOTE, TRAIN_PYTHON, digest, verify_bundle, write_json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "subliminal-forecast-20260929"))
from publish_general_migration import SSH


def remote(command, program=None, timeout=60):
    result = subprocess.run(SSH + [command], input=program, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("publish", "prepare-data", "preview", "dispatch", "status", "sync"))
    p.add_argument("--approved-copy", help="Remote directory containing an approved LIMA copy")
    args = p.parse_args()
    bundle, _ = verify_bundle(ROOT)
    if args.action == "publish":
        names = list(bundle["files_sha256"]) + ["BUNDLE.json"]
        payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in names}
        program = "FILES=" + repr(payload) + "\nROOT=" + repr(REMOTE) + "\n" + r'''
import base64,hashlib,json,fcntl
from pathlib import Path
root=Path(ROOT); decoded={n:base64.b64decode(v) for n,v in FILES.items()}
bundle=json.loads(decoded['BUNDLE.json'])
guard=Path('/projects/u6oz/yuhe/jlens-single-claim-benchmarks-20260911-r1-ops-tqa-cap64-20260915/scripts/isambard/submit.py')
with Path('/projects/u6oz/.jlens-subliminal-submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert hashlib.sha256(guard.read_bytes()).hexdigest()==bundle['guard_sha256'],'Shared guard changed'
 for name,data in decoded.items():
  path=(root/name).resolve()
  assert path.is_relative_to(root.resolve()),'Unsafe publication path'
  if name!='BUNDLE.json':assert hashlib.sha256(data).hexdigest()==bundle['files_sha256'][name]
  if name.endswith('.py'):compile(data,name,'exec')
  if path.exists():assert path.read_bytes()==data,'Published source is immutable: '+name
 for name,data in decoded.items():
  path=root/name
  if not path.exists():
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:f.write(data)
 print(json.dumps({'status':'published_verified','root':str(root),'files':len(decoded),'bundle_sha256':hashlib.sha256(decoded['BUNDLE.json']).hexdigest()}))
'''
        result = json.loads(remote(shlex.quote(TRAIN_PYTHON) + " -", program))
        write_json(ROOT / "PUBLICATION.json", result); print(json.dumps(result)); return
    if args.action in {"prepare-data", "preview", "dispatch"}:
        script = "prepare_data.py" if args.action == "prepare-data" else "dispatch.py"
        command = [TRAIN_PYTHON, REMOTE + "/" + script]
        if args.action == "dispatch": command.append("--submit")
        if args.approved_copy:
            if args.action != "prepare-data": raise ValueError("Approved copy is only a data preparation option")
            command += ["--approved-copy", args.approved_copy]
        print(remote(shlex.join(command)), end=""); return
    program = "ROOT=" + repr(REMOTE) + "\nEXPECTED=" + repr(digest(ROOT / "BUNDLE.json")) + "\n" + r'''
import base64,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT)
assert hashlib.sha256((root/'BUNDLE.json').read_bytes()).hexdigest()==EXPECTED,'Remote bundle changed'
files={};hashes={}
for folder in ['models','data','training','forecasts','slurm']:
 for path in (root/folder).rglob('*'):
  if not path.is_file():continue
  # Keep remote model weights, optimizer state, and tokenized dataset local to Isambard.
  if path.suffix not in {'.json','.jsonl','.out','.err'}:continue
  if folder=='models' and len(path.relative_to(root).parts)>2:continue
  if path.stat().st_size>10*1024*1024:continue
  name=str(path.relative_to(root));data=path.read_bytes()
  files[name]=base64.b64encode(data).decode();hashes[name]=hashlib.sha256(data).hexdigest()
queue=subprocess.run(['squeue','--me','--format=%.18i %.45j %.12T %.10M %R'],capture_output=True,text=True,check=True)
print(json.dumps({'queue':queue.stdout,'files':files,'files_sha256':hashes,'bundle_sha256':EXPECTED}))
'''
    capture = json.loads(remote(shlex.quote(TRAIN_PYTHON) + " -", program))
    files = capture.pop("files")
    if args.action == "sync":
        for name, encoded in files.items():
            data = base64.b64decode(encoded, validate=True)
            import hashlib
            if hashlib.sha256(data).hexdigest() != capture["files_sha256"][name]:
                raise ValueError("Remote file hash differs")
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT.resolve()): raise ValueError("Unsafe mirror path")
            if path.exists() and path.read_bytes() != data and name.endswith("complete.json"):
                raise ValueError("A saved completion receipt changed")
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        write_json(ROOT / "SYNC_RECEIPT.json", capture)
    print(capture["queue"])
    receipts = [n for n in capture["files_sha256"] if n.endswith("complete.json")]
    print(json.dumps({"completion_receipts": receipts, "file_count": len(files)}))


if __name__ == "__main__":
    main()
