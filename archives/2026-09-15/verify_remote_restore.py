"""Read back a complete model from uploaded private release assets; source untouched."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlparse
import zipfile

WORK=Path('/data/code/ouyanghao/oyh/acs_archive_20260915')
class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        result=super().redirect_request(req,fp,code,msg,headers,newurl)
        if urlparse(req.full_url).hostname!=urlparse(newurl).hostname:
            result.remove_header('Authorization')
        return result

token=json.load(sys.stdin)['token']
manifest=json.loads((WORK/'manifest.partial.json').read_text())
file=manifest['files'][0]
needed={manifest['chunks'][k]['pack'] for k in file['chunks']}
packs={p['name']:p for p in json.loads((WORK/'packs.json').read_text())}
directory=WORK/'restore_check';directory.mkdir(exist_ok=True)
opener=urllib.request.build_opener(SafeRedirect())
for name in needed:
    info=packs[name];path=directory/name
    if path.exists():raise FileExistsError(path)
    request=urllib.request.Request(info['url'],headers={'Authorization':'Bearer '+token,
        'Accept':'application/octet-stream','User-Agent':'authorized-archive-restore-check'})
    h=hashlib.sha256();size=0
    with opener.open(request,timeout=180) as response,path.open('xb') as f:
        while True:
            b=response.read(1024**2)
            if not b:break
            f.write(b);h.update(b);size+=len(b)
    if size!=info['size'] or h.hexdigest()!=info['sha256']:raise ValueError('Downloaded pack hash mismatch')
    print('REMOTE_PACK_DOWNLOADED_AND_VERIFIED',name,flush=True)
h=hashlib.sha256();size=0;opened={}
try:
    for key in file['chunks']:
        p=manifest['chunks'][key]['pack']
        if p not in opened:opened[p]=zipfile.ZipFile(directory/p)
        data=opened[p].read(key)
        if hashlib.sha256(data).hexdigest()!=key:raise ValueError('Chunk mismatch')
        h.update(data);size+=len(data)
finally:
    for z in opened.values():z.close()
if h.hexdigest()!=file['sha256'] or size!=file['size']:raise ValueError('Restored model hash mismatch')
result={'file':file['path'],'size':size,'sha256':h.hexdigest(),'github_download_and_complete_model_reconstruction_passed':True,
        'source_files_deleted':0,'scope':'One full model plus all archive packs individually checked before upload and by GitHub digest.'}
(WORK/'restore_verification.json').write_text(json.dumps(result,indent=2))
print('REMOTE_MODEL_RESTORE_VERIFIED',json.dumps(result),flush=True)
