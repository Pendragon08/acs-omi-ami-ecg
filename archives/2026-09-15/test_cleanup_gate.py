"""Isolated cleanup gate tests with temporary data and fake GitHub metadata."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('cleanup',Path(__file__).with_name('cleanup_verified_models.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class CleanupGateTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);module.ROOT=base/'project';module.WORK=base/'work'
        module.WORK.mkdir();(module.ROOT/'results').mkdir(parents=True)
        self.source=module.ROOT/'results/model.pt';self.source.write_bytes(b'test model bytes only')
        self.other=module.ROOT/'results/patient_table.csv';self.other.write_bytes(b'retained-test')
        stat=self.source.stat();h=module.digest(self.source)
        manifest={'release_id':1,'files':[{'path':'results/model.pt','size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':h}],
                  'packs':[{'id':2,'size':12,'sha256':'pack-digest'}]}
        path=module.WORK/'model-manifest.json';path.write_text(json.dumps(manifest))
        self.complete={'manifest_sha256':module.digest(path),'files':1,'manifest_asset_id':3}
        (module.WORK/'ARCHIVE_COMPLETE.json').write_text(json.dumps(self.complete))
        (module.WORK/'restore_verification.json').write_text(json.dumps({'github_download_and_complete_model_reconstruction_passed':True}))
        self.good_digest=True
    def response(self,request,timeout):
        url=request.full_url
        if '/contents/' in url:obj={'content':base64.b64encode(json.dumps(self.complete).encode()).decode()}
        elif '/assets?' in url:obj=[{'id':2,'state':'uploaded','size':12,'digest':'sha256:'+('pack-digest' if self.good_digest else 'wrong')},
                                    {'id':3,'digest':'sha256:'+self.complete['manifest_sha256']}]
        else:obj={'private':True}
        return io.BytesIO(json.dumps(obj).encode())
    def invoke(self,execute=False):
        with patch.object(module.urllib.request,'urlopen',self.response),patch.object(sys,'stdin',io.StringIO('{"token":"unused-test"}')),patch.object(sys,'argv',['test']+(['--execute'] if execute else [])),contextlib.redirect_stdout(io.StringIO()):
            module.main()
    def test_dry_run_preserves_everything(self):
        self.invoke();self.assertTrue(self.source.exists());self.assertTrue(self.other.exists())
    def test_mismatched_remote_keeps_source(self):
        self.good_digest=False
        with self.assertRaises(ValueError):self.invoke(True)
        self.assertTrue(self.source.exists())
    def test_modified_source_kept(self):
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.invoke(True)
        self.assertTrue(self.source.exists())
    def test_execute_only_verified_model(self):
        self.invoke(True);self.assertFalse(self.source.exists());self.assertTrue(self.other.exists())
        self.assertTrue((module.WORK/'cleanup_receipt.json').exists())

if __name__=='__main__':unittest.main()
