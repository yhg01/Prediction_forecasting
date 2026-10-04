"""Publish, submit, and copy the separate Qwen3 seed 1 forecast repeat."""
import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
from common import REMOTE, TRAIN_PYTHON, digest, write_json
from remote import remote
from rerun_qwen3_submit import RUN_ID, verify_repeat

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("publish", "preview", "submit", "status", "sync"))
    args = parser.parse_args()
    bundle = verify_repeat(ROOT)
    out = ROOT / "reruns" / RUN_ID
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = {"action": args.action, "at": stamp, "bundle_sha256": digest(ROOT / "RERUN_QWEN3_SEED1.json")}
    try:
        if args.action == "publish":
            names = list(bundle["files_sha256"]) + ["RERUN_QWEN3_SEED1.json"]
            payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in names}
            program = "ROOT=" + repr(REMOTE) + "\nFILES=" + repr(payload) + "\n" + r'''
import base64,json,hashlib,sys
from pathlib import Path
root=Path(ROOT);sys.path.insert(0,str(root))
from finish_analysis import verify_analysis
verify_analysis(root)
decoded={n:base64.b64decode(v) for n,v in FILES.items()}
bundle=json.loads(decoded['RERUN_QWEN3_SEED1.json'])
for name,data in decoded.items():
 path=(root/name).resolve()
 assert path.is_relative_to(root.resolve()),'Unsafe path'
 if name!='RERUN_QWEN3_SEED1.json':assert hashlib.sha256(data).hexdigest()==bundle['files_sha256'][name]
 if name.endswith('.py'):compile(data,name,'exec')
 if path.exists():assert path.read_bytes()==data,'Published source differs: '+name
for name,expected in bundle['inputs_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,'Original input changed: '+name
for name,data in decoded.items():
 path=root/name
 if not path.exists():
  with path.open('xb') as stream:stream.write(data)
 if name.endswith('.sh'):path.chmod(0o755)
print(json.dumps({'status':'published_verified','files':len(decoded)}))
'''
            receipt["result"] = json.loads(remote(shlex.quote(TRAIN_PYTHON) + " -", program))
        elif args.action in ("preview", "submit"):
            command = [TRAIN_PYTHON, REMOTE + "/rerun_qwen3_submit.py"]
            if args.action == "submit":
                command.append("--submit")
            receipt["result"] = json.loads(remote(shlex.join(command)))
        else:
            program = "ROOT=" + repr(REMOTE) + "\nRUN_ID=" + repr(RUN_ID) + "\nCOPY=" + repr(args.action == "sync") + "\n" + r'''
import base64,json,hashlib,subprocess
from pathlib import Path
folder=Path(ROOT)/'reruns'/RUN_ID
files={};hashes={}
for path in folder.rglob('*'):
 if path.is_file() and path.suffix in {'.json','.jsonl','.out','.err'}:
  name=str(path.relative_to(folder));data=path.read_bytes()
  hashes[name]=hashlib.sha256(data).hexdigest()
  if COPY:files[name]=base64.b64encode(data).decode()
proof=folder/'forecasts/qwen3_8b/seed1/chatml/complete.json'
queue=subprocess.run(['squeue','--me','--format=%.18i %.45j %.12T %.10M %R'],capture_output=True,text=True,check=True)
print(json.dumps({'queue':queue.stdout,'complete':json.loads(proof.read_text()) if proof.exists() else None,'files':files,'files_sha256':hashes}))
'''
            capture = json.loads(remote(shlex.quote(TRAIN_PYTHON) + " -", program))
            files = capture.pop("files")
            if args.action == "sync":
                for name in sorted(files, key=lambda n: n.endswith("complete.json")):
                    path = (out / name).resolve()
                    if not path.is_relative_to(out.resolve()):
                        raise ValueError("Unsafe mirror path")
                    data = base64.b64decode(files[name], validate=True)
                    if hashlib.sha256(data).hexdigest() != capture["files_sha256"][name]:
                        raise ValueError("Copied file hash differs")
                    if path.exists() and path.read_bytes() != data and name.endswith("complete.json"):
                        raise ValueError("A saved completion receipt changed")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".sync.tmp")
                    temporary.write_bytes(data)
                    temporary.replace(path)
            receipt["result"] = capture
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        write_json(out / "dispatch-records" / (stamp + ".json"), receipt)
        raise
    write_json(out / "dispatch-records" / (stamp + ".json"), receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
