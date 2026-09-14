"""Remove transfer fragments only after exact equality with retained pretrain.pt."""
import hashlib
import json
from pathlib import Path
import shutil
BASE=Path('/data/code/ouyanghao/oyh/ACS-OMI-AMI/checkpoints/ecg_lfm')
WORK=Path('/data/code/ouyanghao/oyh/acs_archive_20260915')
complete=BASE/'pretrain.pt';folder=BASE/'parts_20260827'
if BASE.resolve()!=BASE or folder.resolve()!=folder:raise ValueError('Unexpected symlink')
parts=[folder/f'part_{i:02d}' for i in range(32)]
if not all(p.is_file() and not p.is_symlink() for p in parts):raise ValueError('Missing or symlinked transfer part')
if sum(p.stat().st_size for p in parts)!=complete.stat().st_size:raise ValueError('Size mismatch')
before=shutil.disk_usage(BASE).free
joined=hashlib.sha256();rows=[]
for p in parts:
    h=hashlib.sha256();stat=p.stat()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);joined.update(block)
    rows.append({'path':str(p),'sha256':h.hexdigest(),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
whole=hashlib.sha256();stat=complete.stat()
with complete.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):whole.update(block)
if whole.digest()!=joined.digest():raise ValueError('Concatenated parts differ from complete model')
if complete.stat().st_size!=stat.st_size or complete.stat().st_mtime_ns!=stat.st_mtime_ns:raise ValueError('Complete model changed')
out=WORK/'duplicate_parts_cleanup.json'
if out.exists():raise FileExistsError(out)
intent={'retained_model':str(complete),'retained_sha256':whole.hexdigest(),'verified_concatenation':True,'files':rows,'free_before':before,'deleted':False}
(WORK/'duplicate_parts_intent.json').write_text(json.dumps(intent,indent=2))
for row in rows:
    p=Path(row['path']);current=p.stat()
    if p.resolve()!=p or current.st_size!=row['size'] or current.st_mtime_ns!=row['mtime_ns']:raise ValueError('Part changed')
    p.unlink()
intent.update(deleted=True,freed_file_bytes=sum(r['size'] for r in rows),free_after=shutil.disk_usage(BASE).free)
intent['observed_free_space_increase']=intent['free_after']-before
out.write_text(json.dumps(intent,indent=2))
print('DUPLICATE_TRANSFER_PARTS_REMOVED',len(rows),intent['freed_file_bytes'],whole.hexdigest(),flush=True)
