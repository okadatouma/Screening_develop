"""Keep failed attempts immutable; retry under a new folder with the SAME config."""
import fcntl
import json
from pathlib import Path
import time
from .store import json_save,digest


def resolve(root,name):
    root=Path(root);path=root/'retries.json'
    records=json.loads(path.read_text()) if path.exists() else []
    attempts=[r for r in records if r['original']==name]
    return root/(attempts[-1]['folder'] if attempts else name)


def reserve(root,case):
    root=Path(root)
    with (root/'retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=root/'retries.json';records=json.loads(manifest.read_text()) if manifest.exists() else []
        index=1+sum(r['original']==case['name'] for r in records)
        name=case['name']+f'__retry_{index:02d}'
        if (root/name).exists():raise FileExistsError(root/name)
        records.append(dict(original=case['name'],folder=name,attempt=index,created=time.time(),
                            config_hash=digest(json.dumps(case['config'],sort_keys=True).encode()),
                            reason='explicit retry under new source/config-identical run; old failure and source snapshot retained'))
        json_save(manifest,records,replace=manifest.exists())
        return root/name
