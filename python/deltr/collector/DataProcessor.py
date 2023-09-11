from typing import *
from jsonargparse import CLI
from seutil import LoggingUtils
import seutil as su

from deltr.Macros import Macros
from deltr.collector.diff_utils import (
    compute_code_diffs,
    remove_keep_span,
    compute_unique_edits,
    compute_minimal_comment_diffs,
    compute_minimal_replace_diffs,
    format_minimal_diff_spans,
    EDIT_TOKENS,
    EDIT_START_TOKENS,
)

logger = su.log.get_logger(__name__, su.LoggingUtils.INFO)


class DataProcessor:
    SPLITS = ["train", "valid", "test"]

    def meta_edit_data_process(
        self,
        exp: str,
        src_lang: str = "java",
        tgt_lang: str = "cs",
        model_name: str = "metaEdits",
        setting: str = "time-segmented",
    ):
        """
        Process dataset for meta edit model.
        """

        model_data_dir = Macros.data_dir / model_name / setting / exp
        su.io.mkdir(model_data_dir)

        raw_data_dir = Macros.data_dir / "raw" / setting

        for split in self.SPLITS:
            data_list = su.io.load(raw_data_dir / f"delta-translation-{split}.jsonl")

            model_input_file = model_data_dir / f"{split}.{exp}.src"
            model_output_file = model_data_dir / f"{split}.{exp}.tgt"
            model_ground_truth_file = model_data_dir / f"{split}.{exp}.seq"

            model_inputs, model_outputs, golds = [], [], []

            for i, dt in enumerate(data_list):
                # src lang
                src_lang_edits_tokens = compute_minimal_replace_diffs(
                    dt[f"{src_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                    dt[f"{src_lang}-new"]["tokenized_code"].split() + ["<eos>"],
                )[0]
                assert (
                    format_minimal_diff_spans(
                        dt[f"{src_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                        src_lang_edits_tokens,
                    )
                    == dt[f"{src_lang}-new"]["tokenized_code"] + " <eos>"
                ), f"index {i} {dt[f'{src_lang}-old']['tokenized_code']} \n {dt[f'{src_lang}-new']['tokenized_code']}"

                src_lang_edits = " ".join(src_lang_edits_tokens)
                tgt_lang_edits_tokens = compute_minimal_replace_diffs(
                    dt[f"{tgt_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                    dt[f"{tgt_lang}-new"]["tokenized_code"].split() + ["<eos>"],
                )[0]
                assert (
                    format_minimal_diff_spans(
                        dt[f"{tgt_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                        tgt_lang_edits_tokens,
                    )
                    == dt[f"{tgt_lang}-new"]["tokenized_code"] + " <eos>"
                ), f"index {i} {dt[f'{tgt_lang}-old']['tokenized_code']} \n {dt[f'{tgt_lang}-new']['tokenized_code']}"

                tgt_lang_edits = " ".join(tgt_lang_edits_tokens)
                meta_edits, _, _ = compute_code_diffs(
                    src_lang_edits_tokens, tgt_lang_edits_tokens
                )
                meta_edits_plan = remove_keep_span(meta_edits)
                src_lang_target = " ".join(
                    dt[f"{src_lang}-new"]["tokenized_code"].split()
                )
                tgt_lang_source = " ".join(
                    dt[f"{tgt_lang}-old"]["tokenized_code"].split()
                )
                tgt_lang_target = " ".join(
                    dt[f"{tgt_lang}-new"]["tokenized_code"].split()
                )
                golds.append(tgt_lang_target)
                model_inputs.append(
                    f"{src_lang_edits} </s> {tgt_lang_source} </s> {src_lang_target}"
                )
                model_outputs.append(f"{meta_edits_plan} <s> {tgt_lang_edits}")
            # endfor
            su.io.dump(model_input_file, model_inputs, su.io.Fmt.txtList)
            su.io.dump(model_output_file, model_outputs, su.io.Fmt.txtList)
            su.io.dump(model_ground_truth_file, golds, su.io.Fmt.txtList)

    def edit_translation_data_process(
        self,
        exp: str,
        model_name: str = "edit-translation",
        src_lang: str = "java",
        tgt_lang: str = "cs",
        setting: str = "time-segmented",
    ):
        """
        Process dataset for edit translation model.
        """

        model_data_dir = Macros.data_dir / model_name / setting / exp
        su.io.mkdir(model_data_dir)
        raw_data_dir = Macros.data_dir / "raw" / setting

        for split in self.SPLITS:
            data_list = su.io.load(raw_data_dir / f"delta-translation-{split}.jsonl")

            model_input_file = model_data_dir / f"{split}.{exp}.src"
            model_output_file = model_data_dir / f"{split}.{exp}.tgt"
            model_ground_truth_file = model_data_dir / f"{split}.{exp}.seq"

            model_inputs, model_outputs, golds = [], [], []

            for dt in data_list:
                # src lang
                src_lang_edits_tokens = compute_minimal_replace_diffs(
                    dt[f"{src_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                    dt[f"{src_lang}-new"]["tokenized_code"].split() + ["<eos>"],
                )[0]
                assert (
                    format_minimal_diff_spans(
                        dt[f"{src_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                        src_lang_edits_tokens,
                    )
                    == dt[f"{src_lang}-new"]["tokenized_code"] + " <eos>"
                ), f"{dt[f'{src_lang}-old']['tokenized_code']} \n {dt[f'{src_lang}-new']['tokenized_code']}"
                src_lang_edits = " ".join(src_lang_edits_tokens)

                tgt_lang_edits_tokens = compute_minimal_replace_diffs(
                    dt[f"{tgt_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                    dt[f"{tgt_lang}-new"]["tokenized_code"].split() + ["<eos>"],
                )[0]
                assert (
                    format_minimal_diff_spans(
                        dt[f"{tgt_lang}-old"]["tokenized_code"].split() + ["<eos>"],
                        tgt_lang_edits_tokens,
                    )
                    == dt[f"{tgt_lang}-new"]["tokenized_code"] + " <eos>"
                ), f"{dt[f'{tgt_lang}-old']['tokenized_code']} \n {dt[f'{tgt_lang}-new']['tokenized_code']}"
                tgt_lang_edits = " ".join(tgt_lang_edits_tokens)

                src_lang_target = " ".join(
                    dt[f"{src_lang}-new"]["tokenized_code"].split()
                )
                tgt_lang_source = " ".join(
                    dt[f"{tgt_lang}-old"]["tokenized_code"].split()
                )
                tgt_lang_target = " ".join(
                    dt[f"{tgt_lang}-new"]["tokenized_code"].split()
                )
                golds.append(tgt_lang_target)
                assert tgt_lang_source != tgt_lang_target

                model_inputs.append(
                    f"{src_lang_edits} </s> {tgt_lang_source} </s> {src_lang_target}"
                )
                model_outputs.append(f"{tgt_lang_edits}")
            # endfor
            su.io.dump(model_input_file, model_inputs, su.io.Fmt.txtList)
            su.io.dump(model_output_file, model_outputs, su.io.Fmt.txtList)
            su.io.dump(model_ground_truth_file, golds, su.io.Fmt.txtList)


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(DataProcessor, as_positional=False)
