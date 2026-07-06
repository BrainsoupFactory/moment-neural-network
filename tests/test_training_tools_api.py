# -*- coding: utf-8 -*-
import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import TensorDataset

from mnn.utils import training_tools
from mnn.utils import training_tools_api as api
from mnn.utils.training_tools import functional as legacy_functional
from mnn.utils.training_tools import general_prepare as legacy_prepare
from mnn.utils.training_tools import general_train as legacy_train


class TrainingToolsApiTest(unittest.TestCase):
    def test_exports_and_legacy_package_still_work(self):
        self.assertIs(api.TrainProcessCollections, api.TrainProcessCollections)
        self.assertIs(training_tools.make_dataloader, legacy_prepare.make_dataloader)
        self.assertIs(training_tools.RecordMethods, legacy_functional.RecordMethods)
        self.assertIs(training_tools.TrainProcessCollections, legacy_train.TrainProcessCollections)

    def test_source_does_not_import_legacy_training_modules(self):
        source = inspect.getsource(api)
        self.assertNotIn("from . import general_prepare", source)
        self.assertNotIn("from . import functional", source)
        self.assertNotIn("from . import general_train", source)

    def test_config_read_and_reset_match_legacy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, "w") as file:
                yaml.safe_dump({"lr": 0.2, "epochs": 3}, file)
            args = SimpleNamespace(config=config_path, LR_SCHEDULER={"x": 1}, OPTIMIZER={"y": 2}, DATASET={}, DATALOADER={})
            legacy_args = SimpleNamespace(**args.__dict__)
            new_args = api.set_config2args(args)
            expected = legacy_prepare.set_config2args(legacy_args)
            self.assertEqual(new_args.lr, expected.lr)
            self.assertEqual(new_args.epochs, expected.epochs)
            self.assertIsNone(new_args.LR_SCHEDULER)
            self.assertIsNone(new_args.OPTIMIZER)

    def test_transform_compose_matches_legacy_type(self):
        params = {"aug_order": ["ToTensor"]}
        self.assertEqual(type(api.make_transforms_compose(params)), type(legacy_prepare.make_transforms_compose(params)))

    def test_input_predict_score_helpers(self):
        mean = torch.ones(2, 3)
        covariance = torch.eye(3).expand(2, 3, 3)
        preprocess = api.InputPreprocess(dtype="double", mask_mean=True)
        output_mean, output_covariance = preprocess((mean, covariance))
        self.assertEqual(output_mean.dtype, torch.float64)
        torch.testing.assert_close(output_mean, torch.zeros_like(output_mean))
        torch.testing.assert_close(output_covariance, covariance.to(torch.float64))

        logits = torch.tensor([[0.1, 0.9], [2.0, 1.0]])
        prediction = api.PredictMethods.max_mean_predictor(logits)
        torch.testing.assert_close(prediction, torch.tensor([1, 0]))
        self.assertEqual(api.ScoreMethods.equal_protocol(prediction, torch.tensor([1, 0])), 2)

    def test_average_and_progress_meter(self):
        meter = api.AverageMeter("Loss", ":.2f")
        meter.update(2.0, n=2)
        meter.update(4.0, n=2)
        self.assertEqual(meter.avg, 3.0)
        progress = api.ProgressMeter(5, [meter], prefix="Epoch: [0]")
        self.assertIn("Epoch: [0]", progress.display(1))

    def test_dataloader_optimizer_and_criterion_smoke(self):
        dataset = TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        args = SimpleNamespace(bs=2, distributed=False, pin_mem=False, workers=0)
        loader = api.make_dataloader(dataset, args, train=True)
        batch = next(iter(loader))
        self.assertEqual(batch[0].shape, torch.Size([2, 3]))

        model = torch.nn.Linear(3, 2)
        opt_args = SimpleNamespace(lr=0.01, OPTIMIZER=None, LR_SCHEDULER=None, distributed=False)
        optimizer, scheduler = api.prepare_optimizer_scheduler(model.parameters(), opt_args)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNone(scheduler)

        criterion = api.make_criterion({"source": "pytorch", "name": "CrossEntropyLoss"})
        self.assertIsInstance(criterion, torch.nn.CrossEntropyLoss)

    def test_train_and_validate_smoke(self):
        dataset = TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        model = torch.nn.Linear(3, 2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = SimpleNamespace(
            use_cuda=False,
            flatten_input=True,
            scale_factor=1.0,
            input_prepare=None,
            CLIP_PARAMS=None,
            max_grad_norm=None,
            max_grad_value=None,
            task_type="classification",
            distributed=False,
            print_freq=100,
            local_rank=0,
            log_path=None,
        )
        train_process = api.TrainProcessCollections()
        train_process.train_one_epoch(loader, model, criterion, optimizer, 0, args)
        score = train_process.validate(loader, model, criterion, args)
        self.assertGreaterEqual(score, 0.0)

    def test_docs_migration_files_are_indexed(self):
        with open("docs/index.rst") as file:
            index = file.read()
        for path in ("legacy_nn_migration", "legacy_models_snn_migration", "legacy_training_tools_migration"):
            self.assertIn(path, index)
            self.assertTrue(os.path.exists(os.path.join("docs", f"{path}.md")))


if __name__ == "__main__":
    unittest.main()
