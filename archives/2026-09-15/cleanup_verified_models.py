"""Delete ONLY immutable model files with a complete, verified remote backup.

Requires archive_complete.json already committed to the archive branch. Source
files are hashed again before any deletion. Raw data/results tables are excluded.
"""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT=Path('/data/code/ouyanghao/oyh/ACS-OMI-AMI')
WORK=Path('/data/code/ouyanghao/oyh/acs_archive_20260915')
REPO='Pendragon08/acs-omi-ami-ecg'
BRANCH='archive/server-results-20260915'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()

def main():
    token=json.load(sys.stdin)['token']
    def api(path):
        req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,
            headers={'Authorization':'Bearer '+token,'User-Agent':'verified-archive-cleanup','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
    if not api('')['private']:raise RuntimeError('Repository privacy changed')
    complete=json.loads((WORK/'ARCHIVE_COMPLETE.json').read_text())
    manifest_path=WORK/'model-manifest.json'
    if digest(manifest_path)!=complete['manifest_sha256']:raise ValueError('Manifest mismatch')
    manifest=json.loads(manifest_path.read_text())
    if len(manifest['files'])!=complete['files']:raise ValueError('Incomplete file list')
    receipt=json.loads((WORK/'restore_verification.json').read_text())
    if not receipt['github_download_and_complete_model_reconstruction_passed']:raise ValueError('Restore test absent')
    content=api('contents/archives/2026-09-15/archive_complete.json?ref='+BRANCH)
    saved=json.loads(base64.b64decode(content['content']))
    if saved['manifest_sha256']!=complete['manifest_sha256']:raise ValueError('Branch completion record differs')
    assets=[];page=1
    while True:
        rows=api(f'releases/{manifest["release_id"]}/assets?per_page=100&page={page}')
        assets.extend(rows)
        if len(rows)<100:break
        page+=1
    lookup={x['id']:x for x in assets}
    for pack in manifest['packs']:
        found=lookup[pack['id']]
        if found['state']!='uploaded' or found['size']!=pack['size'] or found.get('digest')!='sha256:'+pack['sha256']:
            raise ValueError('Remote pack no longer matches')
    asset=lookup[complete['manifest_asset_id']]
    if asset.get('digest')!='sha256:'+complete['manifest_sha256']:raise ValueError('Remote manifest differs')
    if (WORK/'cleanup_receipt.json').exists() or (WORK/'cleanup_deleted.jsonl').exists():
        raise FileExistsError('Cleanup already started; inspect receipt before retrying')
    verified=[];before=shutil.disk_usage(ROOT).free
    for item in manifest['files']:
        path=ROOT/item['path'];resolved=path.resolve(strict=True)
        if resolved!=path or ROOT/'results' not in resolved.parents or path.suffix not in ('.pt','.pth','.ckpt'):
            raise ValueError('Unsafe target path')
        stat=path.stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns'] or digest(path)!=item['sha256']:
            raise ValueError('Source changed; no deletion: '+item['path'])
        verified.append((path,item,stat.st_ino))
    intent={'files':len(verified),'bytes':sum(x[1]['size'] for x in verified),'manifest_sha256':complete['manifest_sha256'],
            'remote_branch':BRANCH,'remote_assets_verified':True,'source_full_hashes_verified':True,'free_before':before}
    (WORK/'cleanup_intent.json').write_text(json.dumps(intent,indent=2))
    if '--execute' not in sys.argv:
        print('VERIFIED_DRY_RUN',json.dumps(intent));return
    deleted=[]
    with (WORK/'cleanup_deleted.jsonl').open('x') as log:
        for path,item,inode in verified:
            stat=path.stat()
            if path.resolve()!=path or stat.st_ino!=inode or stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
                raise ValueError('Source changed after verification')
            path.unlink()
            entry={'path':item['path'],'bytes':item['size'],'sha256':item['sha256']}
            log.write(json.dumps(entry)+'\n');log.flush();deleted.append(entry)
    result=dict(intent,deleted_files=len(deleted),deleted_bytes=sum(x['bytes'] for x in deleted),
                free_after=shutil.disk_usage(ROOT).free,retained='raw ECG, patient-level tables, logs, code, configs, downloaded checkpoints')
    result['observed_free_space_increase']=result['free_after']-before
    (WORK/'cleanup_receipt.json').write_text(json.dumps(result,indent=2))
    print('VERIFIED_SOURCE_MODELS_REMOVED',json.dumps(result),flush=True)

if __name__=='__main__':main()
