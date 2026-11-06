from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lake_workbench.models.experiments import experimental_models, input_variant


class ExperimentalModelLabelTests(unittest.TestCase):
    def test_input_sources_distinguish_two_five_channel_layouts(self):
        self.assertEqual(input_variant(dict(input_bands=['blue','green','red','nir','swir1'],input_indices=[0,1,2,3,4])),'五光谱')
        self.assertEqual(input_variant(dict(input_bands=['red','green','blue','nir','slope_degrees'])),'RGB-NIR-SLOPE')
        self.assertEqual(input_variant(dict(input_bands=['blue','green','red','nir','swir1','slope_degrees'])),'五光谱+坡度')

    def test_encoder_labels_include_source_and_checkpoint_validation_iou(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'outputs/encoder_test/runs/seed_42/B_overlap_grouped'
            folder.mkdir(parents=True)
            config=dict(encoder='efficientnet-b0',in_channels=5,seed=42,lr=.001,epochs=2,
                arm='B_overlap_grouped',input_bands=['red','green','blue','nir','slope_degrees'])
            (folder/'config.json').write_text(json.dumps(config))
            history=[dict(epoch=1,validation=dict(iou=.981234,loss=.1)),dict(epoch=2,validation=dict(iou=.9,loss=.05))]
            (folder/'history.json').write_text(json.dumps(history))
            for filename,epoch in [('best.pt',1),('best_loss.pt',2)]:
                torch.save(dict(config=config,epoch=epoch,history=history[:epoch]),folder/filename)
            with patch('lake_workbench.models.experiments.PROJECT_ROOT',root):
                items=experimental_models('default')
            best=next(item for item in items if item['weight']=='best.pt')
            loss=next(item for item in items if item['weight']=='best_loss.pt')
            self.assertIn('efficientnet-b0 · RGB-NIR-SLOPE · B',best['label'])
            self.assertTrue(best['label'].endswith('最高验证IoU 98.12%'))
            self.assertEqual(best['selected_validation_iou'],.981234)
            self.assertIn('最低验证损失',loss['label'])
            self.assertTrue(loss['label'].endswith('最低验证损失 90.00%'))
            self.assertNotIn('最高验证IoU',loss['label'])
            self.assertEqual(loss['selected_validation_iou'],.9)
            self.assertEqual(loss['max_validation_iou'],.981234)
            self.assertEqual(best['max_validation_iou'],.981234)


if __name__=='__main__':unittest.main()
