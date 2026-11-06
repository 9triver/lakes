from __future__ import annotations

import tempfile
import json
import base64
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from lake_workbench.models.comparison import _cache_path, _label_path, aligned_seed_labels, ignored_mask_preview, predict_tiled, reference_mask_preview, saved_comparisons, scored_metrics
from PIL import Image
from affine import Affine
from lake_workbench.models.features import assemble_features, feature_names
from lake_workbench.models.runtime import init_torch


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        init_torch()

    def test_rgb_nir_slope_uses_correct_physical_features(self):
        image=np.stack([np.full((3,4),index,dtype='float32') for index in range(5)])
        names=feature_names({'input_bands':['red','green','blue','nir','slope_degrees']},5)
        result=assemble_features(image,names,np.full((3,4),12,dtype='float32'))
        np.testing.assert_array_equal(result[:,1,2],[2,1,0,3,12])
        with self.assertRaisesRegex(ValueError,'坡度'):
            assemble_features(image,names)
        with self.assertRaisesRegex(ValueError,'通道定义'):
            feature_names({},6)

    def test_metrics_exclude_ignored_and_invalid_pixels(self):
        truth=np.array([[1,0,1],[0,255,1]],dtype='uint8')
        prediction=np.array([[1,1,0],[0,1,0]],dtype=bool)
        valid=np.array([[1,1,1],[1,1,0]],dtype=bool)
        value=scored_metrics(prediction,truth,valid)
        self.assertEqual([value[k] for k in ['tp','fp','fn','tn','scored_pixels']],[1,1,1,1,4])
        self.assertAlmostEqual(value['iou'],1/3)
        self.assertIsNone(scored_metrics(np.zeros((2,2),bool),np.zeros((2,2)),np.ones((2,2),bool))['iou'])

    def test_reference_mask_displays_actual_water_background_and_ignore(self):
        truth=np.array([[1,0,255,1]],dtype='uint8')
        valid=np.array([[1,1,1,0]],dtype=bool)
        url=reference_mask_preview(truth,valid)
        image=np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))))
        np.testing.assert_array_equal(image[0],[[61,211,130],[8,12,17],[130,130,130],[24,30,38]])

    def test_ignore_overlay_highlights_255_but_not_water_background_or_invalid_pixels(self):
        truth=np.array([[1,0,255,255]],dtype='uint8')
        valid=np.array([[1,1,1,0]],dtype=bool)
        url=ignored_mask_preview(truth,valid)
        rgba=np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))))
        np.testing.assert_array_equal(rgba[0,:,3],[0,0,180,0])
        np.testing.assert_array_equal(rgba[0,2,:3],[168,85,247])

    def test_overlap_fusion_counts_native_pixels_once_and_preserves_normalization(self):
        generator=np.random.default_rng(7)
        image=generator.normal(size=(5,317,389)).astype('float32')
        network=torch.nn.Conv2d(5,1,1).eval()
        mean=np.arange(5,dtype='float32');std=np.arange(1,6,dtype='float32')
        model=SimpleNamespace(model=network,device=torch.device('cpu'),mean=mean,std=std)
        valid=np.ones(image.shape[1:],bool);valid[:8,:9]=False
        tiled=predict_tiled(model,image,valid,batch_size=3)
        with torch.no_grad():
            expected=torch.sigmoid(network(torch.from_numpy(((image-mean[:,None,None])/std[:,None,None])[None]))).numpy()[0,0]
        np.testing.assert_allclose(tiled[valid],expected[valid],atol=2e-7)
        self.assertEqual(tiled.shape,valid.shape)
        metric=scored_metrics(tiled>=.5,expected>=.5,valid)
        self.assertEqual(metric['scored_pixels'],int(valid.sum()))
        self.assertEqual(metric['fp']+metric['fn'],0)

    def test_reference_must_match_exact_image_date(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'S2_20250101_Swater.shp').touch()
            self.assertIsNone(_label_path(root/'S2_20250201.tif'))
            self.assertEqual(_label_path(root/'S2_20250101.tif'),root/'S2_20250101_Swater.shp')

    def test_seed_reference_preserves_ignore_and_excludes_outside_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'seed.npz'
            transform=Affine(10,0,100,0,-10,200)
            np.savez(path,labels=np.array([[1,255],[0,1]],dtype='uint8'),valid=np.array([[1,1],[1,0]]),
                transform=np.array(tuple(transform)[:6]),crs=np.asarray('EPSG:32648'))
            result=aligned_seed_labels(path,(3,3),transform,'EPSG:32648')
            np.testing.assert_array_equal(result,[[1,255,255],[0,255,255],[255,255,255]])
            metric=scored_metrics(np.ones((3,3),bool),result,np.ones((3,3),bool))
            self.assertEqual(metric['scored_pixels'],2)
            self.assertEqual(metric['tp'],1)
            self.assertEqual(metric['fp'],1)

    def test_saved_results_require_current_weight_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            weight=root/'best.pt';weight.write_bytes(b'old weights')
            catalog=SimpleNamespace(region=SimpleNamespace(key='gansu'))
            site=SimpleNamespace(site_id='gansu_21666')
            with patch('lake_workbench.models.comparison.PROJECT_ROOT',root), patch('lake_workbench.models.comparison._grid',return_value=('fixed-context',)):
                cache=_cache_path('gansu',site.site_id,'fixed-context',weight,.5)
                cache.parent.mkdir(parents=True)
                cache.write_text(json.dumps(dict(context_id='fixed-context',model=dict(key='old-key'))))
                result=saved_comparisons(catalog,site,[('authorized-key',weight)])
                self.assertEqual(len(result['items']),1)
                self.assertTrue(result['items'][0]['result']['cached'])
                self.assertEqual(result['items'][0]['result']['model']['key'],'authorized-key')
                self.assertEqual(saved_comparisons(catalog,site,[],threshold=.5)['items'],[])
                self.assertEqual(saved_comparisons(catalog,site,[('authorized-key',weight)],threshold=.6)['items'],[])
                weight.write_bytes(b'new, different weights')
                self.assertEqual(saved_comparisons(catalog,site,[('authorized-key',weight)])['items'],[])


if __name__=='__main__':
    unittest.main()
