"""Restore byte-exact model files from the manifest and downloaded ZIP packs.

Download private release assets with `gh release download` first. No torch/pickle load.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def restore(manifest,packs,output,only=None):
    data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if data['format']!='sha256-zip-chunks-v1':raise ValueError('Unknown archive format')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    selected=[f for f in data['files'] if only is None or f['path']==only]
    if not selected:raise ValueError('No matching file')
    needed={data['chunks'][k]['pack'] for f in selected for k in f['chunks']}
    for pack in data['packs']:
        if pack['name'] not in needed:continue
        p=Path(packs)/pack['name'];h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        if p.stat().st_size!=pack['size'] or h.hexdigest()!=pack['sha256']:raise ValueError('Pack hash mismatch')
    cache={}
    try:
        for file in selected:
            destination=(root/file['path']).resolve()
            if root not in destination.parents:raise ValueError('Path outside restore root')
            if destination.exists():raise FileExistsError(destination)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.name+'.restoring')
            if temporary.exists():raise FileExistsError(temporary)
            h=hashlib.sha256();size=0
            with temporary.open('xb') as stream:
                for key in file['chunks']:
                    info=data['chunks'][key];name=info['pack']
                    if name not in cache:cache[name]=zipfile.ZipFile(Path(packs)/name)
                    block=cache[name].read(key)
                    if len(block)!=info['size'] or hashlib.sha256(block).hexdigest()!=key:raise ValueError('Chunk mismatch')
                    stream.write(block);h.update(block);size+=len(block)
            if size!=file['size'] or h.hexdigest()!=file['sha256']:raise ValueError('Restored file mismatch')
            temporary.replace(destination)
            print('RESTORED',file['path'],file['sha256'],flush=True)
    finally:
        for archive in cache.values():archive.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--packs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--only')
    a=p.parse_args();restore(a.manifest,a.packs,a.output,a.only)
