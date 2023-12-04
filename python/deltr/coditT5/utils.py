import os
import datetime
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from tqdm import tqdm
import torch
import numpy as np
from jsonargparse.typing import Path_dc, Path_drw, Path_dw, Path_fc, Path_fr
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import (
    LR_SCHEDULER_REGISTRY,
    OPTIMIZER_REGISTRY,
    LightningArgumentParser,
    LightningCLI,
)
import pytorch_lightning as pl
from recordclass import RecordClass


from seutil.LoggingUtils import LoggingUtils
import seutil as su


logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)


class DefaultLightningCLI(LightningCLI):
    def __init__(
        self,
        *args,
        optimizers: Optional[
            List[Tuple[Optional[Union[Type, List[Type]]], str, str]]
        ] = None,
        lr_schedulers: Optional[
            List[Tuple[Optional[Union[Type, List[Type]]], str, str]]
        ] = None,
        prediction_writer: Optional[Callback] = None,
        **kwargs,
    ):
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.prediction_writer = prediction_writer
        kwargs.setdefault("save_config_overwrite", True)
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--exp_dir",
            required=True,
            help="Path to experiment directory",
            type=Union[Path_drw, Path_dc],
        )

        parser.add_argument(
            "--resume",
            required=False,
            help="When training, what to do if a checkpoint already exists: unset (default) = error; True = resume; False = remove (all existing checkpoints)",
            type=bool,
        )

        parser.add_argument(
            "--ckpt_name",
            required=False,
            help="The checkpoint file name to load (under regular ckpt directory); if unset, the latest checkpoint will be loaded",
            type=str,
        )

        parser.add_argument(
            "--no_compute_metrics",
            required=False,
            help="When predicting, do not compute metrics and only collect predictions",
            type=bool,
            default=True,
        )

        parser.add_argument(
            "--no_ckpt_ok",
            required=False,
            help="When predicting, what to do if no checkpoint exists: False (default) = error; True = predict from scratch",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--output_dir",
            required=False,
            help="Path to the output directory during prediction",
            type=Path_dc,
        )

        parser.add_lightning_class_args(ModelCheckpoint, "ckpt")
        parser.set_defaults(
            {
                "ckpt.save_last": True,
                "ckpt.verbose": True,
            }
        )

        if self.optimizers is not None:
            for types, nested_key, link_to in self.optimizers:
                if types is None:
                    types = OPTIMIZER_REGISTRY.classes
                parser.add_optimizer_args(types, nested_key, link_to)

        if self.lr_schedulers is not None:
            for types, nested_key, link_to in self.lr_schedulers:
                if types is None:
                    types = LR_SCHEDULER_REGISTRY.classes
                parser.add_lr_scheduler_args(types, nested_key, link_to)

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config = self.config[self.config["subcommand"]]
        # In ddp mode, default disable find_unused_parameters
        if config["trainer"]["strategy"] == "ddp":
            config["trainer"]["strategy"] = pl.plugins.DDPPlugin(
                find_unused_parameters=False,
            )

        # # Don't save config in non-fit mode
        if self.config["subcommand"] != "fit":
            self.save_config_callback = None

        # Set up experiment directory and logger
        exp_dir = Path(config["exp_dir"].abs_path).resolve()

        config["trainer"]["default_root_dir"] = os.path.relpath(exp_dir, Path.cwd())
        ckpt_dir = exp_dir / "model"
        su.io.mkdir(ckpt_dir)
        config["ckpt"]["dirpath"] = os.path.relpath(ckpt_dir, Path.cwd())

        # locate checkpoint file
        if config["ckpt_path"] is None:
            if config["ckpt_name"] is not None:
                ckpt_file = ckpt_dir / config["ckpt_name"]
            else:
                ckpt_file = self.locate_ckpt(ckpt_dir, self.config["subcommand"])
        else:
            ckpt_file = Path(os.path.abspath(config["ckpt_path"])).resolve()

        if self.config["subcommand"] == "fit":
            # If a checkpoint path is specified, assume we want to resume from it
            if config["ckpt_path"] is not None or config["ckpt_name"] is not None:
                config.setdefault("resume", True)

            # If there is a checkpoint, we must decide what to do with it
            if ckpt_file is not None:
                if config["resume"] is None:
                    raise RuntimeError(
                        f"A checkpoint is present at {ckpt_file}, but I'm not sure what to do with it. Either set `--resume True` to use it or `--resume False` to overwrite it."
                    )
                elif config["resume"] is True:
                    logger.info(f"Resuming from checkpoint {ckpt_file}")
                    config["ckpt_path"] = str(ckpt_file.resolve())
                else:
                    logger.info(f"Removing checkpoints under {ckpt_dir}")
                    su.io.mkdir(ckpt_dir, fresh=True)
                    config["ckpt_path"] = None

        if (
            self.config["subcommand"] == "predict"
            or self.config["subcommand"] == "validate"
            or self.config["subcommand"] == "test"
        ):
            if (
                self.config["subcommand"] == "test"
                or self.config["subcommand"] == "validate"
            ):
                config["trainer"]["gpus"] = 1
            config["model"]["output_dir"] = os.path.relpath(exp_dir, Path.cwd())
            if ckpt_file is not None:
                config["ckpt_path"] = str(ckpt_file.resolve())
                print("Checkpoint path", config["ckpt_path"])
            else:
                if config["no_ckpt_ok"] is False:
                    raise RuntimeError(
                        f"No checkpoint found, cannot predict (unless using `--no_ckpt_ok True` to allow predicting from scratch)"
                    )
                else:
                    logger.info("No checkpoint found, predicting from scratch")

            if self.prediction_writer is None:
                logger.warning(
                    "No prediction writer specified. "
                    "Will not write predictions to disk."
                )
            elif config["model"]["output_dir"] is None:
                logger.warning(
                    "No output directory specified."
                    "Will not write predictions to disk."
                )
            elif self.config["subcommand"] == "predict":
                config["trainer"]["callbacks"].append(
                    self.prediction_writer(
                        config["model"]["output_dir"],
                        config["no_compute_metrics"],
                        config["data"]["dataset"],
                        config["data"]["model"],
                        config["data"]["infer_data"],
                    )
                )

        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
        logger_save_dir = exp_dir / "logs" / self.config["subcommand"]
        logger_version = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        while (logger_save_dir / logger_version).exists():
            time.sleep(1)
            logger_version = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        su.io.mkdir(logger_save_dir)
        config["trainer"]["logger"] = {
            "class_path": "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
            "init_args": {
                "save_dir": os.path.relpath(logger_save_dir, Path.cwd()),
                "name": None,
                # "project": "delta-translation",
                "version": logger_version,
            },
        }

    @classmethod
    def locate_ckpt(cls, ckpt_dir: Path, mode: str) -> Optional[Path]:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if len(ckpt_files) == 0:
            ckpt_file = None
            logger.info(f"No checkpoint found in {ckpt_dir}")
        elif len(ckpt_files) == 1:
            ckpt_file = ckpt_files[0]
            logger.info(f"Found one checkpoint in {ckpt_dir}: {ckpt_file.name}")
        else:
            if (ckpt_dir / "last.ckpt").is_file() and mode == "fit":
                ckpt_file = ckpt_dir / "last.ckpt"
                logger.info(
                    f"Found the last checkpoint in {ckpt_dir}: {ckpt_file.name}"
                )
            else:
                for f in ckpt_files:
                    if f.name == "last.ckpt":
                        ckpt_files.remove(f)
                ckpt_file = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
                logger.warning(
                    f"Multiple checkpoints found in {ckpt_dir}: {[x.name for x in ckpt_files]}; picking the latest modified: {ckpt_file.name}"
                )
        return ckpt_file


class SequenceLabelingDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        source_file_path: Path,
        context_file_path: Path,
        label_file_path: Path,
        tokenizer: Any,
    ):
        """Read data from jsonl files."""
        self.source_code = [
            code.strip()
            for code in open(source_file_path, "r", encoding="utf-8").readlines()
        ]
        self.context = [
            ctx.strip()
            for ctx in open(context_file_path, "r", encoding="utf-8").readlines()
        ]
        self.labels = [
            [int(label) for label in lb.strip().split()]
            for lb in open(label_file_path, "r", encoding="utf-8").readlines()
        ]
        self.tokenized_labels = tokenize_and_align_labels(
            self.source_code, self.labels, tokenizer
        )

    def __len__(self):

        return len(self.source_code)

    def __getitem__(self, index: int):

        return {
            "code": self.source_code[index],
            "context": self.context[index],
            "labels": self.tokenized_labels[index],
        }


class SequenceLabelingChunkDataset(torch.utils.data.Dataset):
    """Dataset for sequence labeling and chunk the data"""

    def __init__(
        self,
        source_file_path: Path,
        context_file_path: Path,
        label_file_path: Path,
        tokenizer: Any,
    ):
        """Read data from jsonl files."""

        self.JAVA_CHUNK_LEN = 240
        self.CS_CHUNK_LEN = 255
        self.tokenizer = tokenizer
        source_code = [
            code.strip()
            for code in open(source_file_path, "r", encoding="utf-8").readlines()
        ]
        context = [
            ctx.strip()
            for ctx in open(context_file_path, "r", encoding="utf-8").readlines()
        ]
        labels = [
            [int(label) for label in lb.strip().split()]
            for lb in open(label_file_path, "r", encoding="utf-8").readlines()
        ]
        tokenized_labels = tokenize_and_align_labels(source_code, labels, tokenizer)
        self.__split_data_to_chunks__(source_code, context, tokenized_labels)

    def __len__(self):

        return len(self.tokenized_code_input)

    def __split_data_to_chunks__(self, source_code, context, tokenized_labels):
        """Split examples into chunks if too long."""

        self.tokenized_code_input = []
        self.tokenized_context_input = []
        self.data_index = []
        self.labels = []
        too_long_context = 0

        for index in tqdm(range(len(source_code)), total=len(source_code)):
            tokenized_code = self.tokenizer.tokenize(source_code[index])
            tokenized_context = self.tokenizer.tokenize(context[index])
            tokenized_label = tokenized_labels[index]
            assert len(tokenized_code) == len(tokenized_labels[index])
            if (
                len(tokenized_code) + len(tokenized_context) + 1
                > self.tokenizer.model_max_length
            ):
                # context_length = min(self.MAX_CTX_LEN, len(tokenized_context))
                # if context_length == self.MAX_CTX_LEN:
                too_long_context += 1
                # start to cut
                code_start_id, code_end_id = 0, 0
                context_start_id, context_end_id = 0, 0
                while code_start_id < len(tokenized_code):
                    code_end_id = self.CS_CHUNK_LEN + code_start_id
                    context_end_id = self.JAVA_CHUNK_LEN + context_start_id
                    self.tokenized_code_input.append(
                        tokenized_code[code_start_id:code_end_id]
                    )
                    self.tokenized_context_input.append(
                        tokenized_context[context_start_id:context_end_id]
                    )
                    self.labels.append(tokenized_label[code_start_id:code_end_id])
                    self.data_index.append(index)
                    code_start_id = code_end_id
                    context_start_id = context_end_id

            else:
                self.tokenized_code_input.append(tokenized_code)
                self.tokenized_context_input.append(tokenized_context)
                self.data_index.append(index)
                self.labels.append(tokenized_label)


        return

    def __getitem__(self, index: int):

        return {
            "code": self.tokenized_code_input[index],
            "context": self.tokenized_context_input[index],
            "labels": self.labels[index],
            "index": self.data_index[index],
        }


def tokenize_and_align_labels(
    source_code: List[str], labels: List[int], tokenizer: Any
) -> List[List[int]]:

    tokenized_labels = []

    for code, label in zip(source_code, labels):
        tokenized_inputs = tokenizer(
            code.split(), is_split_into_words=True, add_special_tokens=False
        )
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_labels.append(label_ids)

    return tokenized_labels


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, source_file_path: Path, target_file_path: Path):
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path
        self.source_offset = []
        self.target_offset = []
        self.n_data = 0

        with open(source_file_path, "rb") as fp:
            self.source_offset = [0]
            while fp.readline():
                self.source_offset.append(fp.tell())
            self.source_offset = self.source_offset[:-1]

        with open(target_file_path, "rb") as fp:
            self.target_offset = [0]
            while fp.readline():
                self.target_offset.append(fp.tell())
            self.target_offset = self.target_offset[:-1]

        assert len(self.target_offset) == len(self.source_offset)

        self.n_data = len(self.target_offset)

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, index: int) -> Tuple:

        if index < 0:
            index = self.n_data + index

        with open(self.source_file_path, "r", errors="replace") as sf, open(
            self.target_file_path, "r", errors="replace"
        ) as tf:
            sf.seek(self.source_offset[index])
            source_line = sf.readline()
            tf.seek(self.target_offset[index])
            target_line = tf.readline()

        return (source_line.strip(), target_line.strip())


class Prediction(RecordClass):
    """Prediction at one data"""

    id: int = -1
    data: str = ""


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, source_file_path: Path, target_file_path: Path):
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path
        self.source_offset = []
        self.target_offset = []
        self.n_data = 0

        with open(source_file_path, "rb") as fp:
            self.source_offset = [0]
            while fp.readline():
                self.source_offset.append(fp.tell())
            self.source_offset = self.source_offset[:-1]

        with open(target_file_path, "rb") as fp:
            self.target_offset = [0]
            while fp.readline():
                self.target_offset.append(fp.tell())
            self.target_offset = self.target_offset[:-1]

        assert len(self.target_offset) == len(self.source_offset)

        self.n_data = len(self.target_offset)

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, index: int) -> Tuple:

        if index < 0:
            index = self.n_data + index

        with open(self.source_file_path, "r", errors="replace") as sf, open(
            self.target_file_path, "r", errors="replace"
        ) as tf:
            sf.seek(self.source_offset[index])
            source_line = sf.readline()
            tf.seek(self.target_offset[index])
            target_line = tf.readline()
        return (source_line.strip(), target_line.strip(), index)
