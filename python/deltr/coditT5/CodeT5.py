import transformers
from transformers import (
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel,
)
from typing import List, Tuple, Dict, Optional, Union, Sequence
from jsonargparse.typing import Path_dc, Path_drw
import os
from pathlib import Path
from seutil import LoggingUtils
import torch
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    instantiate_class,
    SaveConfigCallback,
)
import collections
import numpy as np

from .utils import (
    DefaultLightningCLI,
    ExampleDataset,
    PredictDataset,
    Prediction,
)
from deltr.Macros import Macros
from deltr.eval.evaluate import compute_bleu_scores
from deltr.collector.diff_utils import EDIT_TOKENS

from deltr.coditT5.prediction import PredictionWriter

logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

MAX_LENGTH = 512


class CodeT5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "java2cs",
        model: str = "CodeT5",
        infer_data: str = "test",
        batch_size: int = 2,
        eval_batch_size: int = 8,
    ):
        """
        :model_outputs: {model_name: {train: Path, test: Path}}
        """
        super().__init__()

        pl.seed_everything(42)
        self.data_dir = Macros.data_dir / model / dataset
        self.dataset = dataset
        self.infer_data = infer_data
        self.model = model
        self.save_hyperparameters()
        logger.info(f"Data Module params: \n{self.hparams}")

    def setup(self, stage: Optional[str] = None):
        """Load and encode train/valid/test dataset"""

        self.tokenizer = self.trainer.lightning_module.tokenizer
        self.stage = stage
        if stage == "fit" or stage is None:
            # Process training data
            train_source_file = self.data_dir / f"train.{self.dataset}.src"
            train_target_file = self.data_dir / f"train.{self.dataset}.tgt"
            self.train_dataset = ExampleDataset(train_source_file, train_target_file)

            # Process validatoin data
            valid_source_file = self.data_dir / f"valid.{self.dataset}.src"
            valid_target_file = self.data_dir / f"valid.{self.dataset}.tgt"
            self.valid_dataset = ExampleDataset(valid_source_file, valid_target_file)

        if stage == "predict":
            test_source_file = self.data_dir / f"{self.infer_data}.{self.dataset}.src"
            test_target_file = self.data_dir / f"{self.infer_data}.{self.dataset}.tgt"
            logger.info("Start to process prediction data...")
            self.test_dataset = PredictDataset(test_source_file, test_target_file)

        if stage == "validate":
            valid_source_file = self.data_dir / f"valid.{self.dataset}.src"
            valid_target_file = self.data_dir / f"valid.{self.dataset}.tgt"
            self.valid_dataset = ExampleDataset(valid_source_file, valid_target_file)

    def tokenizer_collate_fn(
        self, batch_data: List[Tuple[str, str]]
    ) -> Sequence[torch.Tensor]:
        """Customize collate function"""
        source_batch = [self.tokenize_sequence(t[0]) for t in batch_data]
        target_batch = [self.tokenize_sequence(t[1]) for t in batch_data]
        max_length = MAX_LENGTH
        batch_size = len(source_batch)

        batched_input_ids, batched_labels_ids = [], []
        for i in range(batch_size):
            batched_input_ids.append(
                self.tokenizer.encode(
                    source_batch[i],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
            )
            batched_labels_ids.append(
                self.tokenizer.encode(
                    target_batch[i],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
            )

        return (
            torch.LongTensor(batched_input_ids),
            torch.LongTensor(batched_labels_ids),
        )

    def tokenize_collate_fn_predict(self, batch_data: List[Tuple[str, str, int]]):
        source_batch = [self.tokenize_sequence(t[0]) for t in batch_data]
        target_batch = [self.tokenize_sequence(t[1]) for t in batch_data]
        index_batch = [t[2] for t in batch_data]
        max_length = MAX_LENGTH
        batch_size = len(source_batch)

        (
            batched_input_ids,
            batched_labels_ids,
        ) = (
            [],
            [],
        )
        for i in range(batch_size):
            batched_input_ids.append(
                self.tokenizer.encode(
                    source_batch[i],
                    max_length=max_length,
                    truncation=True,
                    padding="longest",
                )
            )
            batched_labels_ids.append(
                self.tokenizer.encode(
                    target_batch[i],
                    max_length=max_length,
                    truncation=True,
                    padding="longest",
                )
            )

        return (
            torch.LongTensor(batched_input_ids),
            torch.LongTensor(batched_labels_ids),
            index_batch,
        )

    def tokenize_sequence(self, seq: str) -> List[str]:
        """Given string sequence should be able to be split by space."""

        space_split_tokens = seq.split()
        new_subtokens = []
        for token in space_split_tokens:
            new_subtokens += self.tokenizer.tokenize(" " + token)
        return new_subtokens

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=16,
            collate_fn=self.tokenizer_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=1,
            collate_fn=self.tokenizer_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=0,
            collate_fn=self.tokenizer_collate_fn,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            num_workers=0,
            collate_fn=self.tokenize_collate_fn_predict,
        )


class CodeT5Module(pl.LightningModule):
    # Instantiate the model
    def __init__(
        self,
        pretrained_tokenizer: Union[Path_drw, str],
        pretrained_model: Union[Path_drw, str],
        optimizer_init: dict,
        lr_scheduler_init: dict,
        output_dir=None,
        skip_special_token_when_generate: bool = True,
        beam_size=5,
        num_return_sequences=1,
    ):
        super(CodeT5Module, self).__init__()

        pl.seed_everything(42)
        if isinstance(pretrained_tokenizer, Path_drw):
            pretrained_tokenizer = os.path.relpath(
                Path(pretrained_tokenizer.abs_path), Path.cwd()
            )
        if isinstance(pretrained_model, Path_drw):
            pretrained_model = os.path.relpath(
                Path(pretrained_model.abs_path), Path.cwd()
            )

        self.save_hyperparameters()
        self.beam_size = beam_size
        self.num_return_sequences = num_return_sequences

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.pretrained_model
        )
        self.skip_special_token_when_generate = skip_special_token_when_generate
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Model Module params: \n{self.hparams}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        if "weight_decay" in self.hparams.optimizer_init["init_args"]:
            no_decay = ["bias", "LayerNorm.weight"]
            parameters = [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.optimizer_init["init_args"][
                        "weight_decay"
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            parameters = self.parameters()
        optimizer = instantiate_class(parameters, self.hparams.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler_init)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def training_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        outputs = self.model(
            inputs, labels=labels, attention_mask=attention_masks, return_dict=True
        )
        train_loss = outputs.loss
        self.log_dict({"loss/train": train_loss.item()}, on_step=True)

        return train_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        batch_size = inputs.shape[0]
        outputs = self.model(
            inputs, attention_mask=attention_masks, labels=labels, return_dict=True
        )
        val_loss = outputs.loss
        output_sequences = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_masks,
            num_beams=5,
            num_return_sequences=self.num_return_sequences,
            max_length=MAX_LENGTH,
        )
        pred_sequences = []
        target_sequences = []
        srcs = []
        for input_ids, output_ids, label in zip(inputs, output_sequences, labels):
            pred = self.detokenize(output_ids)
            if pred == "":
                pred = "<EMPTY>"
            target = self.detokenize(label)
            pred_sequences.append(pred)
            target_sequences.append(target)
        _, bleu_score_list = compute_bleu_scores(target_sequences, pred_sequences)
        if self.trainer.datamodule.stage == "validate":
            return pred_sequences
        metrics_list = {"bleu/val": bleu_score_list}
        metrics_list["loss/val"] = [val_loss.item()] * batch_size

        # log the prediction of model
        s = ""
        for i in range(batch_size):
            s += f"# Example {i}\n\n"
            s += f"- gold\n```\n{target_sequences[i]}\n```\n\n"
            s += f"- pred\n```\n{pred_sequences[i]}\n```\n\n"
            s += f"- metrics\n\n"
            for k, v in metrics_list.items():
                s += f"{k}: {v[i]}\n"
            s += "\n"

        self.logger.experiment.add_text("examples/val", s, global_step=self.global_step)
        # self.logger.log_text(
        #     key="validation",
        #     columns=["examples/val"],
        #     data=[[s]],
        #     step=self.global_step,
        # )

        return metrics_list

    def predict_step(self, batch: List[torch.Tensor], batch_idx=-1):
        inputs, labels, indexs = batch
        attention_masks = ~(inputs == self.tokenizer.pad_token_id)
        batch_size = inputs.shape[0]
        pred_sequences = []

        output_sequences = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_masks,
            num_beams=self.beam_size,
            num_return_sequences=self.num_return_sequences,
            max_length=MAX_LENGTH,
        )

        for index, output_ids in zip(indexs, output_sequences):
            pred = self.tokenizer.convert_tokens_to_string(
                self.post_process_edit_sequences(
                    self.tokenizer.convert_ids_to_tokens(
                        output_ids,
                        skip_special_tokens=self.skip_special_token_when_generate,
                    )
                )
            )
            pred_sequences.append(Prediction(index, pred))

        return pred_sequences

    def validation_epoch_end(self, outputs: Union[List[Dict], List[List[str]]]):
        dataset_name = self.trainer.datamodule.dataset
        if self.trainer.datamodule.stage == "validate":
            all_valid_preds = []
            for batch_pred in outputs:
                all_valid_preds.extend(batch_pred)
            output_file = (
                f"valid.{dataset_name}.hyp"
                if self.num_return_sequences == 1
                else f"valid.{dataset_name}.{self.num_return_sequences}.hyp"
            )
            with open(f"{self.hparams.output_dir}/{output_file}", "w") as f:
                for pred in all_valid_preds:
                    f.write(f"{pred}\n")
            return
        metrics_list = collections.defaultdict(list)
        for o in outputs:
            for k in o:
                metrics_list[k] += o[k]
        metrics = summarize_metrics(metrics_list)
        self.log_dict(metrics)

    def detokenize(self, output_ids: torch.Tensor) -> str:
        pred = (
            self.tokenizer.convert_tokens_to_string(
                self.post_process_edit_sequences(
                    self.tokenizer.convert_ids_to_tokens(
                        output_ids,
                        skip_special_tokens=self.skip_special_token_when_generate,
                    )
                )
            )
            .replace("<pad>", "")
            .replace("<s>", "")
            .replace("</s>", "")
        )
        return pred

    def save_pretrained(self, save_dir: Union[str, Path, Path_drw, Path_dc]):
        if isinstance(save_dir, (Path_drw, Path_dc)):
            save_dir = Path(save_dir.abs_path)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def post_process_edit_sequences(self, token_list: List[str]) -> List[str]:
        """Post process token list with edit keywords, manually add space."""
        token_list_after_process = []
        for tk in token_list:
            if tk in self.tokenizer.additional_special_tokens or tk in EDIT_TOKENS:
                token_list_after_process.append(f"Ġ{tk}Ġ")
            else:
                token_list_after_process.append(tk)
        return token_list_after_process


def summarize_metrics(
    metrics: Dict[str, Union[float, List[float]]],
) -> Dict[str, float]:
    metrics_summary = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            metrics_summary[k] = float(np.mean([float(x) for x in v]))
        else:
            metrics_summary[k] = float(v)
    return metrics_summary


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)

    OPTIMIZER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.Optimizer, override=True
    )
    LR_SCHEDULER_REGISTRY.register_classes(
        transformers.optimization, torch.optim.lr_scheduler._LRScheduler, override=True
    )

    DefaultLightningCLI(
        CodeT5Module,
        CodeT5DataModule,
        save_config_callback=SaveConfigCallback,
        prediction_writer=PredictionWriter,
        optimizers=[(None, "optimizer", "model.optimizer_init")],
        lr_schedulers=[(None, "lr_scheduler", "model.lr_scheduler_init")],
    )
