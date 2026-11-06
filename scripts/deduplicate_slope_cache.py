#!/usr/bin/env python3
"""Lossless slope NPY dedup: SHA256 + full byte comparison, atomic hard links."""
import argparse,hashlib,json,os,signal,time,uuid
from pathlib import Path
from lake_workbench.paths import PROJECT_ROOT
from lake_workbench.workspaces.file_lock import process_lock

def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def fingerprint(path):
    with path.open('rb') as f:
        start=os.fstat(f.fileno());h=hashlib.sha256()
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        if signature(start)!=signature(os.fstat(f.fileno())):raise RuntimeError('File changed during hashing')
    return h.hexdigest(),signature(start)
def equal(left,right):
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            x=a.read(4*1024*1024);y=b.read(4*1024*1024)
            if x!=y:return False
            if not x:return True

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    root=PROJECT_ROOT/'data/terrain/aligned_slope';job=PROJECT_ROOT/'outputs/slope_dedup_20261030';job.mkdir(exist_ok=True)
    state=dict(status='running',apply=args.apply,pid=os.getpid(),checked=0,linked=0,already_shared=0,released_bytes=0,errors=[],started_at=time.time())
    def save():
        state['updated_at']=time.time();p=job/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
    with process_lock(PROJECT_ROOT/'data/terrain/.all_site_batch.lock',blocking=False),process_lock(job/'dedup.lock',blocking=False):
        files=sorted(root.glob('*.npy'));state['total']=len(files);save();by_hash={};seen_inodes=set()
        try:
            with (job/'replacements.jsonl').open('a',buffering=1) as journal:
                for p in files:
                    try:
                        stat=p.stat();inode=(stat.st_dev,stat.st_ino)
                        if inode in seen_inodes:
                            state['already_shared']+=1;continue
                        digest,original=fingerprint(p);key=(stat.st_size,digest)
                        if key not in by_hash:
                            by_hash[key]=p;seen_inodes.add(inode);continue
                        canonical=by_hash[key];canonical_stat=canonical.stat()
                        if not equal(canonical,p):raise RuntimeError('SHA256 matches but byte comparison differs')
                        if signature(p.stat())!=original:raise RuntimeError('Duplicate changed before replacement')
                        if args.apply:
                            temporary=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.link')
                            try:
                                os.link(canonical,temporary)
                                if signature(p.stat())!=original or signature(canonical.stat())[:4]!=signature(canonical_stat)[:4]:raise RuntimeError('File changed before atomic link publication')
                                os.replace(temporary,p)
                            finally:temporary.unlink(missing_ok=True)
                            if p.stat().st_ino!=canonical.stat().st_ino:raise RuntimeError('Hard link verification failed')
                            journal.write(json.dumps(dict(path=str(p),canonical=str(canonical),sha256=digest,bytes=stat.st_size,time=time.time()))+'\n')
                        state['linked']+=1
                        if stat.st_nlink==1:state['released_bytes']+=stat.st_blocks*512
                    except Exception as error:
                        state['errors'].append(dict(path=str(p),error=str(error)))
                    finally:
                        state['checked']+=1
                        if state['checked']%100==0:
                            save();print(json.dumps(state),flush=True)
            state['status']='complete' if not state['errors'] else 'partial'
        except BaseException as error:
            state['status']='paused' if isinstance(error,KeyboardInterrupt) else 'failed';state['error']=str(error);raise
        finally:save();print(json.dumps(state),flush=True)
if __name__=='__main__':main()
