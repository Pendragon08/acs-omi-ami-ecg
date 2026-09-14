"""Export scalar aggregate metrics only; leave record-level tables and logs in place."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path('/data/code/ouyanghao/oyh/ACS-OMI-AMI/results')
OUT=Path('/data/code/ouyanghao/oyh/acs_archive_20260915/aggregate_metrics.json')
NAME=re.compile(r'metric|summary|evaluation|report|result|score|statistic',re.I)
SENSITIVE=re.compile(r'patient|subject|record|prediction|probabilities|submission|split|assignment|waveform|raw_',re.I)
METRIC=re.compile(r'auc|auroc|auprc|accuracy|precision|recall|f1|specificity|sensitivity|loss|brier|ece|mcc|reward|epoch|support|threshold',re.I)

def scalars(obj,prefix=''):
    result={}
    if not isinstance(obj,dict):return result
    for key,value in obj.items():
        if SENSITIVE.search(str(key)):continue
        path=prefix+'.'+str(key) if prefix else str(key)
        if isinstance(value,dict):result.update(scalars(value,path))
        elif type(value) in (int,float) and math.isfinite(value) and METRIC.search(path):result[path]=value
    return result

rows=[]
for p in sorted(ROOT.rglob('*.json')):
    if p.is_symlink() or not NAME.search(p.name) or SENSITIVE.search(p.name):continue
    if p.stat().st_size>8*1024**2:continue
    data=p.read_bytes()
    try:metrics=scalars(json.loads(data))
    except (ValueError,UnicodeError):continue
    if metrics:rows.append({'source':str(p.relative_to(ROOT)),'source_sha256':hashlib.sha256(data).hexdigest(),'metrics':metrics})
OUT.write_text(json.dumps({'scope':'Allowlisted finite numeric aggregate metrics; no rows, arrays, free-text values or patient identifiers.',
                          'sources':rows},ensure_ascii=False,indent=2))
print('AGGREGATE_SOURCES',len(rows),'METRICS',sum(len(r['metrics']) for r in rows))
