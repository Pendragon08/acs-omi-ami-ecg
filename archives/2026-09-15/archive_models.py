"""Lossless, content-deduplicated model backup to a PRIVATE GitHub release.

Never loads a PyTorch pickle and never deletes source files. Token via stdin only.
ZIP storage boundaries improve deduplication of frozen backbone tensors.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import urllib.request
import urllib.error
import zipfile

ROOT = Path('/data/code/ouyanghao/oyh/ACS-OMI-AMI')
WORK = Path('/data/code/ouyanghao/oyh/acs_archive_20260915')
REPO = 'Pendragon08/acs-omi-ami-ecg'
BLOCK = 8 * 1024**2
PACK = 256 * 1024**2

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(BLOCK),b''):h.update(b)
    return h.hexdigest()

def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')))
    temp.replace(path)

def pieces(path):
    size=path.stat().st_size
    cuts={0,size}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z, open(path,'rb') as f:
            for info in z.infolist():
                f.seek(info.header_offset)
                header=f.read(30)
                if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP header')
                a,b=struct.unpack_from('<HH',header,26)
                start=info.header_offset+30+a+b
                end=start+info.compress_size
                if not 0<=start<=end<=size:raise ValueError('Invalid ZIP extent')
                cuts.update((start,end))
    with open(path,'rb') as f:
        position=0
        for end in sorted(cuts):
            while position<end:
                data=f.read(min(BLOCK,end-position))
                if not data:raise EOFError(path)
                position+=len(data)
                yield data
        assert position==size

class Archive:
    def __init__(self, credentials):
        self.token=credentials['token'];self.release=int(credentials['release_id'])
        self.chunks={};self.files=[];self.packs=[];self.zip=None;self.raw=0
        self.pack_path=None;self.total_read=0;self.unique=0;self.start=time.time()
        self.pending=[]

    def request(self,url,method='GET',data=None,headers=None):
        h={'Authorization':'Bearer '+self.token,'Accept':'application/vnd.github+json',
           'X-GitHub-Api-Version':'2022-11-28','User-Agent':'authorized-model-backup'}
        h.update(headers or {})
        with urllib.request.urlopen(urllib.request.Request(url,data=data,method=method,headers=h),timeout=600) as r:
            return json.load(r)

    def upload(self,path):
        name=path.name;expected=sha(path);size=path.stat().st_size
        for attempt in range(3):
            try:
                # Every retry checks remote state before transmitting again.
                assets=[];page=1
                while True:
                    part=self.request(f'https://api.github.com/repos/{REPO}/releases/{self.release}/assets?per_page=100&page={page}')
                    assets.extend(part)
                    if len(part)<100:break
                    page+=1
                existing=next((a for a in assets if a['name']==name),None)
                if existing:
                    if existing.get('digest')=='sha256:'+expected and existing['size']==size and existing['state']=='uploaded':return existing
                    raise RuntimeError('Remote asset name conflicts; source retained: '+name)
                with open(path,'rb') as f:
                    result=self.request(f'https://uploads.github.com/repos/{REPO}/releases/{self.release}/assets?name={name}',
                        method='POST',data=f,headers={'Content-Type':'application/octet-stream','Content-Length':str(size)})
                verified=self.request(result['url'])
                if verified.get('digest')!='sha256:'+expected or verified['size']!=size or verified['state']!='uploaded':
                    raise RuntimeError('Remote digest verification failed: '+name)
                return verified
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if attempt==2:raise
                time.sleep(5*(attempt+1))
        raise RuntimeError('Upload failed')

    def flush(self):
        if self.zip is None:return
        self.zip.close();self.zip=None
        # Test all compressed members before upload; actual remote digest checked too.
        with zipfile.ZipFile(self.pack_path) as z:
            for key in self.pending:
                if hashlib.sha256(z.read(key)).hexdigest()!=key:raise ValueError('Local chunk restore mismatch')
        remote=self.upload(self.pack_path)
        self.packs.append({'name':self.pack_path.name,'id':remote['id'],'size':remote['size'],
                           'sha256':remote['digest'].split(':')[1],'url':remote['url']})
        save(WORK/'packs.json',self.packs)
        print('PACK_VERIFIED',self.pack_path.name,remote['size'],'read_GiB',round(self.total_read/1024**3,2),flush=True)
        # Only the verified temporary pack is removed. All source files remain.
        self.pack_path.unlink();self.pending=[];self.raw=0
        self.progress()

    def progress(self):
        save(WORK/'progress.json',{'status':'archiving','completed_files':len(self.files),'verified_packs':len(self.packs),
             'source_bytes_read':self.total_read,'unique_bytes':self.unique,'uploaded_bytes':sum(p['size'] for p in self.packs),
             'elapsed_seconds':time.time()-self.start,'source_files_deleted':0})

    def add(self,data):
        key=hashlib.sha256(data).hexdigest()
        if key in self.chunks:return key
        if self.zip is None:
            if shutil.disk_usage(WORK).free<8*1024**3+PACK:raise RuntimeError('Disk reserve gate')
            self.pack_path=WORK/f'model-pack-{len(self.packs):04d}.zip'
            if self.pack_path.exists():raise FileExistsError(self.pack_path)
            self.zip=zipfile.ZipFile(self.pack_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)
        self.zip.writestr(key,data)
        self.chunks[key]={'pack':self.pack_path.name,'size':len(data)}
        self.pending.append(key);self.raw+=len(data);self.unique+=len(data)
        # First pack validates upload early; later packs remain under 2 GiB.
        threshold=32*1024**2 if not self.packs else PACK
        if self.raw>=threshold:self.flush()
        return key

    def run(self):
        remote=self.request(f'https://api.github.com/repos/{REPO}')
        if not remote['private']:raise RuntimeError('Archive requires a private repository')
        inventory=[]
        for p in sorted((ROOT/'results').rglob('*')):
            if p.is_symlink():continue
            if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt'):
                resolved=p.resolve(strict=True)
                if ROOT/'results' not in resolved.parents:raise ValueError('Outside results')
                stat=p.stat();inventory.append((p,stat.st_size,stat.st_mtime_ns))
        save(WORK/'inventory.json',{'count':len(inventory),'bytes':sum(x[1] for x in inventory),
             'files':[{'path':str(p.relative_to(ROOT)),'size':s,'mtime_ns':m} for p,s,m in inventory]})
        for p,size,mtime in inventory:
            h=hashlib.sha256();keys=[]
            for data in pieces(p):
                h.update(data);self.total_read+=len(data);keys.append(self.add(data))
            current=p.stat()
            if current.st_size!=size or current.st_mtime_ns!=mtime:raise RuntimeError('Source changed during backup')
            self.files.append({'path':str(p.relative_to(ROOT)),'size':size,'mtime_ns':mtime,'sha256':h.hexdigest(),'chunks':keys})
            # Manifest checkpoint makes partial progress recoverable without deleting sources.
            save(WORK/'manifest.partial.json',{'files':self.files,'chunks':self.chunks,'packs':self.packs})
            self.progress()
            print('MODEL_READ',len(self.files),'of',len(inventory),str(p.relative_to(ROOT)),flush=True)
        self.flush()
        manifest={'format':'sha256-zip-chunks-v1','repository':REPO,'release_id':self.release,
                  'source_root':str(ROOT),'scope':'trained model files under results only; no raw ECG or patient tables',
                  'files':self.files,'chunks':self.chunks,'packs':self.packs,'source_files_deleted':0}
        save(WORK/'model-manifest.json',manifest)
        remote=self.upload(WORK/'model-manifest.json')
        save(WORK/'ARCHIVE_COMPLETE.json',{'manifest_asset_id':remote['id'],'manifest_sha256':sha(WORK/'model-manifest.json'),
             'files':len(self.files),'source_bytes':sum(x['size'] for x in self.files),'packs':len(self.packs),
             'uploaded_bytes':sum(x['size'] for x in self.packs),'source_files_deleted':0})
        print('ARCHIVE_COMPLETE_NO_SOURCE_DELETION',flush=True)

if __name__=='__main__':
    credentials=json.load(sys.stdin)
    WORK.mkdir(exist_ok=True)
    if (WORK/'archive.log').exists():raise FileExistsError('Archive already started; inspect state before retrying')
    pid=os.fork()
    if pid:
        print(json.dumps({'pid':pid,'work':str(WORK)}),flush=True);sys.exit(0)
    os.setsid()
    log=os.open(str(WORK/'archive.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(log,1);os.dup2(log,2);os.close(log)
    null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0);os.close(null)
    try:Archive(credentials).run()
    except BaseException as e:
        # Do not serialize credentials, HTTP request objects or response bodies.
        save(WORK/'ARCHIVE_FAILED.json',{'exception_type':type(e).__name__,'message':str(e),'source_files_deleted':0})
        print('ARCHIVE_FAILED',type(e).__name__,str(e),flush=True)
        sys.exit(1)
