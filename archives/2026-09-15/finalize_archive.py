"""Wait for full archive, publish completion, then run hash-gated authorized cleanup."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

WORK=Path('/data/code/ouyanghao/oyh/acs_archive_20260915')
REPO='Pendragon08/acs-omi-ami-ecg'
BRANCH='archive/server-results-20260915'

def main(credentials):
    token=credentials['token']
    def api(path,method='GET',body=None):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request('https://api.github.com/repos/'+REPO+'/'+path,data=data,method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'authorized-archive-finalizer','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
    def publish(name,data,message):
        path='contents/archives/2026-09-15/'+name
        body={'message':message,'branch':BRANCH,'content':base64.b64encode(data).decode()}
        try:
            old=api(path+'?ref='+BRANCH)
            if base64.b64decode(old['content'])==data:return
            body['sha']=old['sha']
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
        api(path,'PUT',body)
    while not (WORK/'ARCHIVE_COMPLETE.json').exists():
        if (WORK/'ARCHIVE_FAILED.json').exists():raise RuntimeError('Archive failed; sources retained')
        time.sleep(30)
    complete=(WORK/'ARCHIVE_COMPLETE.json').read_bytes()
    publish('archive_complete.json',complete,'Verify complete model archive before server cleanup')
    print('COMPLETION_COMMITTED',flush=True)
    if not (WORK/'cleanup_receipt.json').exists():
        result=subprocess.run([sys.executable,str(WORK/'cleanup_verified_models.py'),'--execute'],
                              input=json.dumps({'token':token}),text=True,capture_output=True)
        (WORK/'cleanup_execution.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError('Cleanup gate failed; inspect cleanup_execution.log')
    for name in ('cleanup_receipt.json','cleanup_intent.json','restore_verification.json'):
        publish(name,(WORK/name).read_bytes(),'Record verified server model cleanup')
    receipt=json.loads((WORK/'cleanup_receipt.json').read_text())
    old=api('contents/archives/2026-09-15/README.md?ref='+BRANCH)
    text=base64.b64decode(old['content']).decode()
    text=text.replace('状态：**模型上传中，源模型尚未删除**。',
        '状态：**完整模型归档、远端校验、真实恢复测试和服务器清理均已完成**。')
    text+='\n\n## 清理结果\n\n已按校验清单删除 '+str(receipt['deleted_files'])+' 份训练模型；实际可用空间增加约 %.2f GiB。原始数据、患者级表格、日志、代码与下载checkpoint保留。详见 `archive_complete.json` 与 `cleanup_receipt.json`。\n'%(receipt['observed_free_space_increase']/1024**3)
    publish('README.md',text.encode(),'Document completed model preservation and disk recovery')
    complete_obj=json.loads(complete)
    manifest=json.loads((WORK/'model-manifest.json').read_text())
    api('releases/'+str(manifest['release_id']),'PATCH',{
        'body':'Archive complete and verified. '+str(complete_obj['files'])+' trained model files preserved losslessly. '
               'Server model copies removed only after remote digest checks and a complete-model download/restore test. '
               'Raw ECG and patient-level tables excluded and retained on server. Restore instructions and receipts: '
               'https://github.com/'+REPO+'/tree/'+BRANCH+'/archives/2026-09-15'})
    (WORK/'FINALIZED.json').write_text(json.dumps({'state':'archive_verified_and_cleanup_complete','branch':BRANCH,
                         'deleted_files':receipt['deleted_files'],'freed_bytes':receipt['observed_free_space_increase']},indent=2))
    print('ARCHIVE_AND_CLEANUP_FINALIZED',flush=True)

if __name__=='__main__':
    credentials=json.load(sys.stdin)
    if (WORK/'finalizer.log').exists():raise FileExistsError('Finalizer already started')
    pid=os.fork()
    if pid:print(json.dumps({'pid':pid,'work':str(WORK)}),flush=True);sys.exit(0)
    os.setsid();log=os.open(str(WORK/'finalizer.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.dup2(log,1);os.dup2(log,2);os.close(log)
    null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0);os.close(null)
    try:main(credentials)
    except BaseException as e:
        (WORK/'FINALIZER_FAILED.json').write_text(json.dumps({'exception_type':type(e).__name__,'message':str(e)}))
        print('FINALIZER_FAILED',type(e).__name__,str(e),flush=True);sys.exit(1)
