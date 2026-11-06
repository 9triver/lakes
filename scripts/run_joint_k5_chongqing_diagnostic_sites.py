"""Prepare fixed historical-date full-site inputs on server and run frozen joint K5 evaluation."""
from pathlib import Path
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from PIL import Image
from lake_workbench.imagery.validity import valid_pixel_mask
from lake_workbench.models.comparison import aligned_seed_labels
from lake_workbench.models.features import aligned_slope
import run_joint_k5_slope_server as shared

ROOT=shared.ROOT
OUT=ROOT/'outputs/guizhou_hunan_joint_k5_chongqing_diagnostic_sites_20261106'
PLAN=[('chongqing_24107', 'S2A_MSIL1C_20250720'), ('chongqing_24096', 'S2A_MSIL1C_20250720'), ('chongqing_23285', 'S2B_MSIL1C_20250407'), ('chongqing_23025', 'S2B_MSIL1C_20250407'), ('chongqing_23375', 'S2B_MSIL1C_20250407'), ('chongqing_23374', 'S2B_MSIL1C_20250407')]


def main():
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True)
    start=time.time();completed=[]
    for site,product in PLAN:
        province,number=site.split('_')
        target=OUT/'inputs'/f'{site}.npz';meta_path=OUT/'inputs'/f'{site}.json'
        if target.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert shared.sha(target)==meta['input_sha256']
            completed.append(site);continue
        shared.dump(OUT/'preparation_status.json',dict(status='preparing_reference',site=site,completed=completed,pid=os.getpid()))
        print('CONTEXT',site,product,flush=True)
        url=f'http://127.0.0.1:18765/api/workspaces/default/regions/{province}/sites/{site}/model-comparison/context?'
        url+=urllib.parse.urlencode(dict(asset=site+'_'+product,reference='spectral_osm_intersection'))
        with urllib.request.urlopen(url,timeout=240) as response:context=json.load(response)
        assert context['imagery']['asset']==site+'_'+product
        shared.dump(OUT/'inputs'/f'{site}_context.json',context)
        path=ROOT/'data/regions'/province/'raw/local_imagery'/number/context['imagery']['name']
        label=ROOT/'data/model_comparison_references'/province/'spectral_osm_intersection'/context['reference']['name']
        with rasterio.open(path) as src:
            raw=src.read();valid=valid_pixel_mask(raw,src.nodatavals,src.read_masks())
            truth=aligned_seed_labels(label,valid.shape,src.transform,src.crs)
            transform=list(src.transform)[:6]
        assert int(valid.sum())==context['valid_pixels']
        for value,key in [(0,'background_pixels'),(1,'water_pixels'),(255,'ignored_pixels')]:
            assert int(((truth==value)&valid).sum())==context['reference'][key]
        shared.dump(OUT/'preparation_status.json',dict(status='preparing_slope',site=site,completed=completed,pid=os.getpid()))
        print('SLOPE',site,valid.shape,flush=True)
        slope,terrain=aligned_slope(path,valid)
        preview=np.array(Image.open(io.BytesIO(base64.b64decode(context['preview'].split(',',1)[1]))).convert('RGB'))
        np.savez_compressed(target,optical=raw,slope=slope.astype('float32'),valid=valid,truth=truth,preview=preview,transform=transform)
        meta={k:context[k] for k in ['context_id','region','site_id','valid_pixels','imagery','reference']}
        meta.update(terrain=terrain,input_sha256=shared.sha(target),image_sha256=shared.sha(path),label_sha256=shared.sha(label),
                    source='Server18765 full-site grid, frozen historical date; freshly generated spectral intersection OSM reference. Not original patch-pooled reference.',
                    historical_comparison='Prior difference ranking was patch-union coverage; full-site area and regenerated seed counts are not directly comparable.')
        shared.dump(meta_path,meta);completed.append(site)
        print('PREPARED',site,'bytes',target.stat().st_size,'elapsed',round(time.time()-start,1),flush=True)
        del raw,valid,truth,slope
    shared.dump(OUT/'preparation_status.json',dict(status='prepared',completed=completed,seconds=time.time()-start))
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_joint_k5_site.py'),'--sites',*[s for s,_ in PLAN],
                    '--output',str(OUT)],check=True)
    # The evaluator also supports earlier local frozen bundles; correct this run's provenance explicitly.
    provenance=json.loads((OUT/'protocol.json').read_text())
    provenance['reference']='Server18765 spectral intersection OSM references at selected historical dates; full-site evaluation, not frozen historical patch labels'
    provenance['caveat']='Diagnostic sites intentionally selected for previous input differences. Chongqing was not used to train the joint models; sites selected by historical original U-Net five-optical versus slope test-IoU difference. Exploratory transfer evaluation, not a new blind test; full-site versus old pooled patches differ.'
    shared.dump(OUT/'protocol.json',provenance)
    report=(OUT/'report.md').read_text().replace('沿用此前本地18768冻结影像与光谱∩OSM参考','使用服务器指定历史日期的整site影像与本次冻结的光谱∩OSM参考')
    (OUT/'report.md').write_text(report+'\n\n这些site经历史差异定向挑选；重庆未用于本轮联合模型训练；这些site按历史模型测试IoU差异挑选。本次整site范围与旧patch覆盖范围不同，不能直接混比面积或作为独立泛化测试。\n')


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        shared.dump(OUT/'preparation_status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        raise
