"""Copy and verify the separate repeat analysis and figures."""
import base64
import hashlib
import json
from pathlib import Path
from remote import remote
from common import REMOTE, write_json
from rerun_qwen3_submit import RUN_ID

ROOT = Path(__file__).resolve().parent


def main():
    program = "ROOT=" + repr(REMOTE) + "\nRUN_ID=" + repr(RUN_ID) + "\n" + r'''
import base64,hashlib,json
from pathlib import Path
folder=Path(ROOT)/'reruns'/RUN_ID/'analysis-ci'
proof=folder/'complete.json'
if not proof.exists():
 print(json.dumps({'status':'pending','files':{}}))
else:
 complete=json.loads(proof.read_text());files={}
 for name,expected in complete['files_sha256'].items():
  path=(folder/name).resolve()
  assert path.is_relative_to(folder.resolve()),'Unsafe path'
  data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==expected,'Analysis artifact differs'
  files[name]=base64.b64encode(data).decode()
 files['complete.json']=base64.b64encode(proof.read_bytes()).decode()
 print(json.dumps({'status':'completed','files':files}))
'''
    capture = json.loads(remote("/projects/u6oz/yuhe/jlens_belief-v0/.venv/bin/python -", program))
    files = capture.pop("files")
    folder = ROOT / "reruns" / RUN_ID / "analysis-ci"
    if files:
        proof = json.loads(base64.b64decode(files["complete.json"], validate=True))
        for name in sorted(files, key=lambda n: n == "complete.json"):
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe mirror path")
            data = base64.b64decode(files[name], validate=True)
            if name != "complete.json" and hashlib.sha256(data).hexdigest() != proof["files_sha256"][name]:
                raise ValueError("Copied analysis hash differs")
            if path.exists() and path.read_bytes() != data:
                raise ValueError("A saved repeat analysis artifact changed")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".sync.tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        write_json(folder / "SYNC_RECEIPT.json", {"status": "completed", "files_sha256": proof["files_sha256"]})
    print(json.dumps({"status": capture["status"], "files_copied": len(files)}))


if __name__ == "__main__":
    main()
