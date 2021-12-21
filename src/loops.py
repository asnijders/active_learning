# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden

import flash
from flash.core.data.utils import _STAGES_PREFIX
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_5_0, requires
from flash.core.utilities.stages import RunningStage
from flash.image.classification.integrations.baal.data import ActiveLearningDataModule
from flash.image.classification.integrations.baal.dropout import InferenceMCDropoutTask

if not _PL_GREATER_EQUAL_1_5_0:
    from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
else:
    from pytorch_lightning.trainer.connectors.data_connector import _DataLoaderSource

from src.strategies import select_acquisition_fn


class ActiveLearningLoop(Loop):

    def __init__(self, config, label_epoch_frequency: int, inference_iteration: int = 2, should_reset_weights: bool = True):
        """The `ActiveLearning Loop` describes the following training procedure. This loop is connected with the
        `ActiveLearningTrainer`
        Example::
            while unlabelled data or budget critera not reached:
                if labelled data
                    trainer.fit(model, labelled data)
                if unlabelled data:
                    predictions = trainer.predict(model, unlabelled data)
                    uncertainties = heuristic(predictions)
                    request labellelisation for the sample with highest uncertainties under a given budget
        Args:
            label_epoch_frequency: Number of epoch to train on before requesting labellisation.
            inference_iteration: Number of inference to perform to compute uncertainty.
        """
        super().__init__()
        self.label_epoch_frequency = label_epoch_frequency
        self.inference_iteration = inference_iteration
        self.should_reset_weights = should_reset_weights
        self.fit_loop: Optional[FitLoop] = None
        self.progress = Progress()
        self._model_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._lightning_module: Optional[flash.Task] = None
        self.acquisition_fn = select_acquisition_fn(fn_id=config.acquisition_fn)
        self.config = config

    @property
    def done(self) -> bool:
        return self.progress.current.completed >= self.max_epochs

    def connect(self, fit_loop: FitLoop):
        self.fit_loop = fit_loop
        self.max_epochs = self.fit_loop.max_epochs
        self.fit_loop.max_epochs = self.label_epoch_frequency

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        # assert isinstance(self.trainer.datamodule, ActiveLearningDataModule)
        self.trainer.predict_loop._return_predictions = True
        self._lightning_module = self.trainer.lightning_module
        self._model_state_dict = deepcopy(self._lightning_module.state_dict())
        # self.inference_model = s

    def reset(self) -> None:
        pass

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """
        This function does the following
        - if there is still labelled data, we reset the dataloaders for the unlabelled train data and for the val data
        - if there is a test set, we reset the test dataloader
        - if there is unlabelled data we reset the dataloader for the unlabelled data
        """
        if self.trainer.datamodule.has_labelled_data:
            self._reset_dataloader_for_stage(RunningStage.TRAINING)
            self._reset_dataloader_for_stage(RunningStage.VALIDATING)
            if self.trainer.datamodule.has_test:
                self._reset_dataloader_for_stage(RunningStage.TESTING)
        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_dataloader_for_stage(RunningStage.PREDICTING)
        self.progress.increment_ready()

    def advance(self, *args: Any, **kwargs: Any) -> None:

        self.progress.increment_started()

        if self.trainer.datamodule.has_labelled_data:  # This statement starts fitting on the labelled data
            self.fit_loop.run()

        if self.trainer.datamodule.has_test:  # if there is test data..
            self._reset_testing()  # reset
            metrics = self.trainer.test_loop.run()
            if metrics:
                self.trainer.logger.log_metrics(metrics[0], step=self.trainer.global_step)

        if self.trainer.datamodule.has_unlabelled_data:  # if there is still unlabelled data
            self._reset_predicting()  # reset model for prediction
            probabilities = self.acquisition_fn.acquire_instances(config=self.config,
                                                                  model=model,
                                                                  dm=dm,
                                                                  k=config.labelling_batch_size)  # obtain new samples

            self.trainer.datamodule.label(probabilities=probabilities)  # label new samples
        else:
            raise StopIteration

        self._reset_fitting()
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        if self.trainer.datamodule.has_unlabelled_data and self.should_reset_weights:
            # reload the weights to retrain from scratch with the new labelled data.
            self._lightning_module.load_state_dict(self._model_state_dict)
        self.progress.increment_completed()
        return super().on_advance_end()

    def on_run_end(self):
        self._reset_fitting()
        self._teardown()
        return super().on_run_end()

    def on_save_checkpoint(self) -> Dict:
        return {"datamodule_state_dict": self.trainer.datamodule.state_dict()}

    def on_load_checkpoint(self, state_dict) -> None:
        self.trainer.datamodule.load_state_dict(state_dict.pop("datamodule_state_dict"))

    def __getattr__(self, key):
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def _connect(self, model: LightningModule):
        if _PL_GREATER_EQUAL_1_5_0:
            self.trainer.training_type_plugin.connect(model)
        else:
            self.trainer.accelerator.connect(model)

    def _reset_fitting(self):
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.trainer.lightning_module.on_train_dataloader()
        self._connect(self._lightning_module)
        self.fit_loop.epoch_progress = Progress()

    def _reset_predicting(self):
        self.trainer.state.fn = TrainerFn.PREDICTING
        self.trainer.predicting = True
        self.trainer.lightning_module.on_predict_dataloader()
        # self._connect(self.inference_model)

    def _reset_testing(self):
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.state.status = TrainerStatus.RUNNING
        self.trainer.testing = True
        self.trainer.lightning_module.on_test_dataloader()
        self._connect(self._lightning_module)

    def _reset_dataloader_for_stage(self, running_state: RunningStage):
        dataloader_name = f"{_STAGES_PREFIX[running_state]}_dataloader"
        # If the dataloader exists, we reset it.
        dataloader = (
            getattr(self.trainer.datamodule, dataloader_name)
            if is_overridden(dataloader_name, self.trainer.datamodule)
            else None
        )
        if dataloader:
            if _PL_GREATER_EQUAL_1_5_0:
                setattr(
                    self.trainer._data_connector,
                    f"_{dataloader_name}_source",
                    _DataLoaderSource(self.trainer.datamodule, dataloader_name),
                )
            else:
                setattr(
                    self.trainer.lightning_module,
                    dataloader_name,
                    _PatchDataLoader(dataloader(), running_state),
                )
            setattr(self.trainer, dataloader_name, None)
            # TODO: Resolve this within PyTorch Lightning.
            try:
                getattr(self.trainer, f"reset_{dataloader_name}")(self.trainer.lightning_module)
            except MisconfigurationException:
                pass

    def _teardown(self) -> None:
        self.trainer.train_dataloader = None
        self.trainer.val_dataloaders = None
        self.trainer.test_dataloaders = None
        self.trainer.predict_dataloaders = None
        # Hack
        self.trainer.lightning_module.train_dataloader = None
        self.trainer.lightning_module.val_dataloader = None
        self.trainer.lightning_module.test_dataloader = None
        self.trainer.lightning_module.predict_dataloader = None


class CustomFitLoop(FitLoop):

    def __init__(self, config, label_epoch_frequency: int, inference_iteration: int = 2, should_reset_weights: bool = True):
        """The `ActiveLearning Loop` describes the following training procedure. This loop is connected with the
        `ActiveLearningTrainer`
        Example::
            while unlabelled data or budget critera not reached:
                if labelled data
                    trainer.fit(model, labelled data)
                if unlabelled data:
                    predictions = trainer.predict(model, unlabelled data)
                    uncertainties = heuristic(predictions)
                    request labellelisation for the sample with highest uncertainties under a given budget
        Args:
            label_epoch_frequency: Number of epoch to train on before requesting labellisation.
            inference_iteration: Number of inference to perform to compute uncertainty.
        """
        super().__init__()
        self.label_epoch_frequency = label_epoch_frequency
        self.inference_iteration = inference_iteration
        self.should_reset_weights = should_reset_weights
        self.fit_loop: Optional[FitLoop] = None
        self.progress = Progress()
        self._model_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._lightning_module: Optional[flash.Task] = None
        self.acquisition_fn = select_acquisition_fn(fn_id=config.acquisition_fn)
        self.config = config

    @property
    def done(self):
        """Provide a condition to stop the loop."""
        return self.trainer.datamodule.has_unlabelled_data is False

    def advance(self):
        """
        Access your dataloader/s in whatever way you want.
        Do your fancy optimization things.
        Call the LightningModule methods at your leisure.
        """

        self.progress.increment_started()

        # first, fit a model from scratch on the seed data
        if self.trainer.datamodule.has_labelled_data:
            self.trainer.datamodule.train.set_mode('L')  # point to labelled data
            self.trainer.reset_train_dataloader()  # reset train dataloader
            self.trainer.lightning_module.reset_weights()  # (re-)init model
            self.trainer.state.fn = TrainerFn.FITTING
            self.trainer.training = True
            self.fit_loop.run()  # fit model on labelled data
            if self.trainer.datamodule.has_dev:
                self.trainer.state.fn = TrainerFn.TESTING
                self.evaluation_loop.run()  # evaluate model on dev set

        # if we have unlabelled data, select new instances for labelling using previously fitted model
        if self.trainer.datamodule.has_unlabelled_data:
            self.trainer.datamodule.train.set_mode('U')  # point to unlabelled data
            self.trainer.reset_train_dataloader()  # reset train dataloader
            self.trainer.state.fn = TrainerFn.PREDICTING
            self.trainer.predicting = True
            # label new instances
            to_be_labelled = self.acquisition_fn.acquire_instances(config=self.config,
                                                                   model=self.trainer.lightning_module,
                                                                   dm=self.trainer.datamodule,
                                                                   k=self.config.labelling_batch_size)

            self.trainer.datamodule.train.set_mode('L')  # re-set pointer to labelled data
            self.trainer.datamodule.train.label_instances(to_be_labelled)  # label new instances and move from U to L

        else:
            raise StopIteration

        self._reset_fitting()
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        if self.trainer.datamodule.has_unlabelled_data and self.should_reset_weights:
            # reload the weights to retrain from scratch with the new labelled data.
            self._lightning_module.reset_weights
        self.progress.increment_completed()
        return super().on_advance_end()


    def _reset_fitting(self):
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.trainer.lightning_module.on_train_dataloader()
        self._connect(self._lightning_module)
        self.fit_loop.epoch_progress = Progress()

