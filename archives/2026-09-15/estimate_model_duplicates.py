"""ZIP-directory estimate only; CRC is never used as backup integrity proof."""
from pathlib import Path
import json
import zipfile
root=Path('/data/code/ouyanghao/oyh/ACS-OMI-AMI/results')
total=0;estimated=0;seen=set();count=0
for p in root.rglob('*.pt'):
    if p.is_symlink() or not p.is_file():continue
    count+=1;total+=p.stat().st_size
    if not zipfile.is_zipfile(p):estimated+=p.stat().st_size;continue
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            k=(i.file_size,i.CRC)
            if k not in seen:estimated+=i.file_size;seen.add(k)
report={'model_files':count,'source_GiB':total/1024**3,'estimated_unique_payload_GiB':estimated/1024**3,
        'warning':'Estimate only; actual backup uses SHA256 and byte-exact restore, never CRC as identity.'}
print(json.dumps(report))
