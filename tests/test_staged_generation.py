"""Only temporary files and mocked image work; never generates regional data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lake_workbench.workspaces.file_lock import process_lock
from lake_workbench.workspaces import WorkspaceStore

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('staged_generator_under_test',ROOT/'scripts/generate_spectral_osm_training_data.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


class FileLockTests(unittest.TestCase):
    def test_reentrant_and_other_process_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'same.lock'
            with process_lock(path),process_lock(path):
                code='from lake_workbench.workspaces.file_lock import process_lock; import sys\ntry:\n with process_lock(sys.argv[1],blocking=False):pass\nexcept BlockingIOError:sys.exit(7)'
                self.assertEqual(subprocess.run([sys.executable,'-c',code,str(path)]).returncode,7)
            with process_lock(path,blocking=False):pass

    def test_concurrent_updates_do_not_lose_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'count').write_text('0')
            code='from pathlib import Path; import sys,time; from lake_workbench.workspaces.file_lock import process_lock\np=Path(sys.argv[1])\nfor _ in range(8):\n with process_lock(p/"lock"):\n  n=int((p/"count").read_text());time.sleep(.002);(p/"count").write_text(str(n+1))'
            jobs=[subprocess.Popen([sys.executable,'-c',code,d]) for _ in range(3)]
            self.assertTrue(all(job.wait()==0 for job in jobs));self.assertEqual((p/'count').read_text(),'24')

    def test_workspace_instances_share_reentrant_lock(self):
        with tempfile.TemporaryDirectory() as d:
            a=WorkspaceStore({},root=Path(d));b=WorkspaceStore({},root=Path(d))
            with a.build_transaction(),b.build_transaction(),a.transaction(),b.transaction():pass


class PipelineTests(unittest.TestCase):
    def test_interrupt_restores_previous_directory(self):
        import os
        from lake_workbench.training.logical_patches import _replace_directory
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'live';new=Path(d)/'staged'
            old.mkdir();new.mkdir();(old/'old').touch();(new/'new').touch()
            original=os.replace;calls=[]
            def interrupted(a,b):
                calls.append((a,b))
                if len(calls)==2:raise KeyboardInterrupt()
                return original(a,b)
            with patch('lake_workbench.training.logical_patches.os.replace',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):_replace_directory(new,old)
            self.assertTrue((old/'old').exists())

    def test_missing_preview_requires_rebuild_even_if_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'preview.png';p.touch()
            r=dict(logical_size='256',grid_stride='128',preview_path=str(p),preview_base_path=str(p),review_status='excluded')
            self.assertTrue(pipeline.patch_complete([r],256,128))
            p.unlink();self.assertFalse(pipeline.patch_complete([r],256,128))

    def test_seed_first_batch_and_resume_original_selection(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'state.json';events=[]
            sites=[SimpleNamespace(site_id=name) for name in ['hunan_1','hunan_2']]
            fake_store=SimpleNamespace(root=Path(d)/'workspace',get=lambda *a,**k:None)
            fake_catalog=SimpleNamespace(get_site=lambda key:next(s for s in sites if s.site_id==key))
            def seed(catalog,store,workspace,site,*args,**kwargs):
                events.append(('seed',site.site_id));return dict(site_id=site.site_id,label_id='seed-'+site.site_id,cache_hit=True)
            def sample(catalog,store,workspace,site,record):return dict(record,sample_id='sample-'+site.site_id)
            def commit(store,args,batch):events.append(('commit',len(batch)));return dict(patches=20,generated_patches=20)
            with patch.multiple(pipeline,load_region_configs=lambda:({'hunan':object()},'hunan'),WorkspaceStore=lambda *a,**k:fake_store,
                    SiteCatalog=lambda *a:fake_catalog,ensure_seed=seed,prepare_sample=sample,patch_index=lambda *a:{},commit_batch=commit):
                self.assertEqual(pipeline.main(['--region','hunan','--sites','hunan_1','hunan_2','--state-file',str(state)]),0)
                self.assertEqual(events[:2],[('seed','hunan_1'),('seed','hunan_2')])
                self.assertEqual([e for e in events if e[0]=='commit'],[('commit',2)])
                events.clear()
                self.assertEqual(pipeline.main(['--region','hunan','--resume',str(state),'--stage','seeds']),0)
                self.assertFalse(any(e[0]=='commit' for e in events))
                self.assertEqual(json.loads(state.read_text())['site_ids'],['hunan_1','hunan_2'])
                events.clear()
                self.assertEqual(pipeline.main(['--region','hunan','--resume',str(state)]),0)
                self.assertEqual([e for e in events if e[0]=='commit'],[('commit',2)])



if __name__=='__main__':unittest.main()
