"""Refresh an explicit owned-job manifest from mlq; never controls queue state."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',type=int,action='append',default=[])
    parser.add_argument('--out',type=Path,default=Path('benchmarks/sepclip_weight_microscopy/jobs.json'))
    args=parser.parse_args()
    old=json.loads(args.out.read_text()) if args.out.exists() else {'jobs':[]}
    ids=set(args.id)|{row['id'] for row in old['jobs']}
    records=[]
    for job_id in sorted(ids):
        row=json.loads(subprocess.check_output(['mlq','show',str(job_id),'--json'],text=True))
        if not row['name'].startswith('sepclip_microscopy_'):
            raise ValueError(f'Not an owned job: {job_id} {row["name"]}')
        if row['maxParallelRuns']!=1:
            raise ValueError(f'Wrong declared limit: {job_id}')
        records.append(row)
    old.update(updated_at=time.time(),declared_limit=1,jobs=records)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps({r['id']:r['state'] for r in records}))


if __name__=='__main__':main()
