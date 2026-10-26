"""Unified research commands. All generated output is written to new paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .workflow import ROOT, benchmark, provenance, write


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('scan');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'results/screening_topology/scan')
    p=sub.add_parser('smoke');p.add_argument('--output',type=Path,default=ROOT/'results/screening_topology/smoke')
    p=sub.add_parser('benchmark');p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('experiments.yaml'));p.add_argument('--output',type=Path,default=ROOT/'results/screening_topology/benchmark');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('report');p.add_argument('--input',type=Path,default=ROOT/'results/screening_topology');p.add_argument('--output',type=Path)
    p=sub.add_parser('train');p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('training.yaml'));p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='scan':
        from screening_analysis.checkpoint_catalog import discover
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'provenance.json',provenance());write(args.output/'inventory.json',discover(args.root));print(args.output)
    elif args.command=='smoke':
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'provenance.json',provenance(20261026))
        command=[sys.executable,'-m','unittest','tests.test_topology','tests.test_topology_run','tests.test_topology_cdef',
                 'tests.test_topology_regularization','tests.test_phase_metrics','tests.test_benchmark_protocol','tests.test_topology_cli','-v']
        with (args.output/'tests.txt').open('x') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        write(args.output/'status.json',dict(returncode=result.returncode,command=command,baseline_tests_modified=False))
        if result.returncode:raise SystemExit(result.returncode)
        print(args.output)
    elif args.command=='benchmark':benchmark(args.manifest,args.output,args.resume)
    elif args.command=='report':
        from .plots import report
        report(args.input,args.output or args.input/'report')
    elif args.command=='train':
        from .train_job import jobs
        jobs(args.manifest,args.output,args.resume,args.dry_run)


if __name__=='__main__':main()
