"""G2b: exact unaffected values versus G1; version complex energies/updates separately."""
import argparse
import json
from pathlib import Path
from tests import baseline
from tests.support import require_numerics
require_numerics()
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('old',type=Path);p.add_argument('new',type=Path)
    args=p.parse_args()
    if args.new.exists():p.error('existing data will not be overwritten')
    arrays,info=baseline.numerical_snapshot()
    counts=baseline.compare(args.old,arrays,info,exact=True,vmc_gradient_version='g2b')
    args.new.mkdir(exist_ok=False)
    np.savez(args.new/'values.npz',**arrays)
    (args.new/'metadata.json').write_text(json.dumps(info,indent=2))
    (args.new/'comparison.json').write_text(json.dumps(dict(version='g2b-v1',old_snapshot=str(args.old),**counts),indent=2))
    print(json.dumps(counts,indent=2))
