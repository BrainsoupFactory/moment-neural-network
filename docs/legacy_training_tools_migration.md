# Legacy `training_tools` Migration

The original `mnn.utils.training_tools` package remains importable for compatibility, but its split implementation files are legacy APIs. New code should use the single-file API in `mnn.utils.training_tools_api`.

The new module consolidates the public classes and functions from `general_prepare.py`, `functional.py`, and `general_train.py` into one explicit import surface. The legacy files are expected to be removed in a future release.

## Migration Table

| Legacy API | New API |
| --- | --- |
| `mnn.utils.training_tools.general_prepare.make_model` | `mnn.utils.training_tools_api.make_model` |
| `mnn.utils.training_tools.general_prepare.prepare_dataloader` | `mnn.utils.training_tools_api.prepare_dataloader` |
| `mnn.utils.training_tools.general_prepare.prepare_optimizer_scheduler` | `mnn.utils.training_tools_api.prepare_optimizer_scheduler` |
| `mnn.utils.training_tools.general_prepare.deploy_config` | `mnn.utils.training_tools_api.deploy_config` |
| `mnn.utils.training_tools.functional.RecordMethods` | `mnn.utils.training_tools_api.RecordMethods` |
| `mnn.utils.training_tools.functional.InputPreprocess` | `mnn.utils.training_tools_api.InputPreprocess` |
| `mnn.utils.training_tools.functional.DistributedOps` | `mnn.utils.training_tools_api.DistributedOps` |
| `mnn.utils.training_tools.general_train.TrainProcessCollections` | `mnn.utils.training_tools_api.TrainProcessCollections` |
| `mnn.utils.training_tools.general_train.general_train_pipeline` | `mnn.utils.training_tools_api.general_train_pipeline` |
| `mnn.utils.training_tools.general_train.general_distributed_train_pipeline` | `mnn.utils.training_tools_api.general_distributed_train_pipeline` |

## Example

Legacy:

```python
from mnn.utils.training_tools import general_train

general_train.general_train_pipeline(args)
```

New:

```python
from mnn.utils import training_tools_api

training_tools_api.general_train_pipeline(args)
```

## Notes

- The old package-level exports such as `mnn.utils.training_tools.make_dataloader` remain available for now.
- The new module keeps the same training pipeline behavior but avoids internal cross-module imports.
- This refactor does not make training tools Torch-only; dataset preparation, seeding, and NumPy-to-Tensor helpers still use NumPy where the original workflow requires it.
