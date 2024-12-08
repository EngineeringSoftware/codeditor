import re
import copy
import os
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import *
import sys
from tqdm import tqdm
from jsonargparse import CLI
from seutil import (
    LoggingUtils,
    IOUtils,
    BashUtils,
    TimeUtils,
    io,
    TimeoutException,
    bash,
)
import difflib
import random

from deltr.collector.DataCollector import (
    projects_map,
    cs_port_date,
    compute_minimal_code_diffs,
)
from deltr.Macros import Macros
from deltr.collector.utils import (
    include_jaccard,
    jaccard,
    get_commit_date,
    tokenize_code,
)
from deltr.Environment import Environment
from deltr.collector.ProjectData import ProjectData

line_comment_pattern = r"//(.*?)\n"
block_comment_pattern = r"/\*(.*?)\*/"

cs_java_type_map = {
    "bool": "boolean",
    "bool?": "Boolean",
    "sbtye": "byte",
    "sbtye?": "Byte",
    "ushort": "short",
    "uint": "int",
    "uint?": "Integer",
    "int?": "Integer",
    "ulong": "long",
    "char?": "Character",
    "list": "ArrayList",
}

"""
1. Collect java and C# aligned diff: collect_aligned_data()  # collect aligned raw diffs from projects
  1.1 build_sha_changed_file_map()
    Build dict: SHA -> [changed file]  # java and c#
  1.2 mine_changed_methods()
    Build dict: SHA -> [changed methods]  # java and c#
  1.3 aggregate_methods_histories()
    Build dict: method_hash -> [method diff]  # java and c#
    NOTE: method_hash = {m_path}.{class_name}.{method_name}-{params}
  1.4 collect_aligned_method_history 
    1.4.0 build_unique_method_id() Build dict: method_id -> path.class_name.method_name-params  # for java and c#
    NOTE: method_id is min_path.file_name.class_name.method_name-params # should be unique
    1) first use the file name.class_name.method_name-params
    2) if the id is not unique, add the prior dir name
    1.4.1 align_project_java_csharp_method() Match java_method_id and cs_method_id
      if java_method_id == cs_method_id --> match
      else # match java params and csharp params by rules / manually
    1.4.2 collect_method_diff_history() Build dict: method_id -> {java: [method diff], c#: [method diff]}=
    1.4.3 filter_aligned_diff()
      Build list: [java diff, c# diff]
      NOTE: 1) use commit date to find paired diffs. i.e. for each java diff, find the closest c# diff in time and distance should be < 90 days
            2) use jaccard sim between add/del tokens and add/del lines
2. Find the exact method from collected diffs
  build_delta_translation_dataset()
3. Split by time (target language time split c#, java)
  problem: only consider commit time
"""


class RealDiffCollector:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

    # Constants
    commit_date_threshold = 90  # days
    token_sim_threshold = 0.4
    overlap_token_sim_threshold = 0.6
    line_sim_threshold = 0.5

    def __init__(self) -> None:
        self.results_dir = Macros.results_dir / "repo-data"
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_results_dir = Macros.repos_results_dir

        # data split params
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1

    def collect_real_diff_for_projects(self):
        for java_proj in projects_map:
            self.collect_real_aligned_diff_for_project(java_project_name=java_proj)

    def collect_real_aligned_diff_for_project(self, java_project_name: str):
        """script for collecting data from real git history for a project"""

        for lang in ["java", "cs"]:  # hard-code for java and c# only
            if lang == "cs":
                target_project_name = projects_map[java_project_name]
            else:
                target_project_name = java_project_name
            self.build_sha_changed_file_map(project_name=target_project_name, lang=lang)
            self.mine_changed_methods(project_name=target_project_name, lang=lang)
            self.aggregate_methods_histories(
                project_name=target_project_name, lang=lang
            )

        self.align_java_csharp_method(java_project_name=java_project_name)
        _ = input(
            "Human should help aligning method id in two projects. Finished? (y/n)"
        )
        self.collect_project_diff_history(java_project_name=java_project_name)
        self.filter_project_aligned_diff(java_project_name=java_project_name)

    # 1.1
    def build_sha_changed_file_map(self, project_name: str, lang: str):
        """
        Build a dictionary where the key is the SHA and value is the list of changed files.
        Do not constrain the files we collect.
        """
        downloads_dir = self.repos_downloads_dir / project_name
        if lang == "java":
            cs_project_name = projects_map[project_name]
            start_date = cs_port_date[cs_project_name]
        else:
            start_date = cs_port_date[project_name]

        # 0. download repos
        project_url = f"git@github.com:{project_name.split('_')[0]}/{project_name.split('_')[1]}.git"
        if not downloads_dir.exists():
            self.logger.info(f"Cloning repo {project_name} ... ")
            with IOUtils.cd(self.repos_downloads_dir):
                try:
                    with TimeUtils.time_limit(300):
                        BashUtils.run(
                            f"git clone {project_url} {project_name}",
                            expected_return_code=0,
                        )
                    # end with
                except TimeoutException:
                    self.logger.info(
                        f"{project_name} exceeds time limit, ignore this one."
                    )
                    return
                except:
                    self.logger.warning(
                        f"Project {project_name} failed: {sys.exc_info()}"
                    )
                    return
            # end with

        # 1. get all shas
        with IOUtils.cd(downloads_dir):
            branch_name = bash.run(
                "git rev-parse --abbrev-ref HEAD", check_returncode=0
            ).stdout.strip()
            BashUtils.run(f"git checkout {branch_name}", expected_return_code=0)
            shalist = BashUtils.run(
                f"git log --color=never --since='{start_date}' --first-parent --no-merges --pretty=format:'%H'"
            ).stdout.split("\n")
            shalist = [sha[:8] for sha in shalist]
        self.logger.info(f"{len(shalist)} commits found")
        shalist = shalist[::-1]  # in chronological order (old SHA before new SHA)

        # 2. Check the changed files in each commit
        commits_to_files = defaultdict(list)
        with IOUtils.cd(downloads_dir):
            for i in tqdm(range(len(shalist) - 1)):
                cur_sha, pre_sha = shalist[i + 1], shalist[i]
                changed_files = BashUtils.run(
                    f"git diff {pre_sha} {cur_sha} --name-only"
                ).stdout.split("\n")
                changed_files = [f for f in changed_files if f.split(".")[-1] == lang]
                # find intersection files
                for cf in changed_files:
                    commits_to_files[f"{pre_sha}-{cur_sha}"].append(cf)
        self.logger.info(f"{len(commits_to_files)} commits remained")
        io.dump(
            self.repos_results_dir / project_name / f"{lang}-commits-to-files.json",
            commits_to_files,
            io.Fmt.jsonNoSort,
        )

    # 1.2
    def mine_changed_methods_for_projects(self):
        for java_proj in projects_map:
            self.logger.info(f"Mining project {java_proj}")
            self.mine_changed_methods(java_proj, "java")
            # self.mine_changed_methods(projects_map[java_proj], "cs")

    def mine_changed_methods(self, project_name: str, lang: str):
        """Mine the changed methods in changed files between two consecutive commits."""

        git_history = io.load(
            self.repos_results_dir / project_name / f"{lang}-commits-to-files.json"
        )
        sha_to_files = defaultdict(set)
        for sha in git_history:
            changed_files = git_history[sha]
            prev_sha, cur_sha = sha.split("-")[0], sha.split("-")[1]
            sha_to_files[cur_sha] = sha_to_files[cur_sha].union(set(changed_files))
            sha_to_files[prev_sha] = sha_to_files[prev_sha].union(set(changed_files))
        # endfor
        repos_results_dir = self.repos_results_dir / project_name
        project_git_changed_methods_history = OrderedDict()
        for sha in tqdm(git_history, total=len(git_history)):
            prev_sha, cur_sha = sha.split("-")[0], sha.split("-")[1]
            prev_changed_files = sha_to_files[prev_sha]
            cur_changed_files = sha_to_files[cur_sha]
            assert len(prev_changed_files) > 0
            assert len(cur_changed_files) > 0
            # 1. collect methods from two shas
            prev_sha, cur_sha = sha.split("-")[0], sha.split("-")[1]
            if not (
                repos_results_dir / "collector" / f"{lang}-method-data-{prev_sha}.json"
            ).exists():
                self.collect_methods_for_commit(
                    project_name, lang, prev_sha, list(prev_changed_files)
                )
            if not (
                repos_results_dir / "collector" / f"{lang}-method-data-{cur_sha}.json"
            ).exists():
                self.collect_methods_for_commit(
                    project_name, lang, cur_sha, list(cur_changed_files)
                )
            # 2. extract changed methods
            changed_methods_list = self.collect_changed_methods_for_commit(
                project_name, lang, prev_sha, cur_sha
            )
            if len(changed_methods_list) > 0:
                project_git_changed_methods_history[
                    f"{prev_sha}-{cur_sha}"
                ] = changed_methods_list
        # end for
        self.logger.info(f"Collect {len(project_git_changed_methods_history)} history.")
        io.dump(
            self.repos_results_dir
            / project_name
            / f"{lang}-changed-methods-in-git-history.json",
            project_git_changed_methods_history,
            io.Fmt.jsonNoSort,
        )

    # 1.3
    def aggregate_methods_histories(self, project_name: str, lang: str):
        """
        Aggregate each methods' change histories
        A dict where key is method name and value is list of dict which represents change.
        """

        repos_results_dir = self.repos_results_dir / project_name
        sha_2_methods = io.load(
            repos_results_dir / f"{lang}-changed-methods-in-git-history.json"
        )
        project_git_changed_methods_history = defaultdict(list)
        for sha in tqdm(sha_2_methods, total=len(sha_2_methods)):
            changed_methods = sha_2_methods[sha]
            prev_sha, cur_sha = sha.split("-")[0], sha.split("-")[1]
            for m in changed_methods:
                old_m = m[0]
                new_m = m[1]
                m_path = old_m["path"]
                class_name = old_m["class_name"]
                method_name = old_m["name"]
                params = str(old_m["params"])
                method_hash = f"{m_path}.{class_name}.{method_name}-{params}"
                new_method_hash = f"{new_m['path']}.{class_name}.{method_name}-{params}"

                # get diff
                d = difflib.Differ()
                if "".join(new_m["code"].split()) == "".join(old_m["code"].split()):
                    continue
                sha1_diff = tokenize_code(" ".join(old_m["code"].split()))
                sha2_diff = tokenize_code(" ".join(new_m["code"].split()))
                added_tokens, deled_tokens = compute_minimal_code_diffs(
                    sha1_diff, sha2_diff
                )

                old_code_lines: List[str] = [
                    " ".join(cl.split()) + "\n"
                    for cl in old_m["code"].splitlines(keepends=True)
                ]
                new_code_lines: List[str] = [
                    " ".join(cl.split()) + "\n"
                    for cl in new_m["code"].splitlines(keepends=True)
                ]
                code_diff = list(d.compare(old_code_lines, new_code_lines))
                del_code = [code for code in code_diff if code[0] == "-"]
                add_code = [code for code in code_diff if code[0] == "+"]
                project_git_changed_methods_history[method_hash].append(
                    {
                        "old-sha": prev_sha,
                        "old-method-hash": method_hash,
                        "new-method-hash": new_method_hash,
                        "new-sha": cur_sha,
                        "add-tokens": added_tokens,
                        "del-tokens": deled_tokens,
                        "add-code": add_code,
                        "del-code": del_code,
                    }
                )
        self.logger.info(
            f"{len(project_git_changed_methods_history)} {lang} methods found in {project_name}'s git history."
        )
        io.dump(
            self.repos_results_dir
            / project_name
            / f"{lang}-method-change-history.json",
            project_git_changed_methods_history,
            io.Fmt.jsonNoSort,
        )

    def build_unique_method_id(
        self, java_changed_methods_file: Path, cs_changed_methods_file: Path
    ):
        """Build a map for java and c# hash to unique id"""

        java_change_history = io.load(java_changed_methods_file)
        cs_change_history = io.load(cs_changed_methods_file)
        # 0. collect unique java method id
        java_id_map = defaultdict(list)
        for java_hash in java_change_history:
            java_method_id = java_hash.split("/")[-1].replace(".java", "").lower()
            java_id_map[java_method_id].append(java_hash)
        # endfor
        new_java_hash_map = {}
        for java_method_id, java_hash_list in java_id_map.items():
            if len(java_hash_list) > 1:
                smallest_index = -1
                for index in range(1, 20):
                    dir_index = -1 * index
                    java_id_set = set()
                    for java_hash in java_hash_list:
                        new_hash = (
                            ".".join(java_hash.split("/")[dir_index:])
                            .replace(".java", "")
                            .lower()
                        )
                        java_id_set.add(new_hash)
                    # endfor
                    if len(java_id_set) == len(java_hash_list):
                        smallest_index = dir_index
                        break
                for java_hash in java_hash_list:
                    new_hash = (
                        ".".join(java_hash.split("/")[smallest_index:])
                        .replace(".java", "")
                        .lower()
                    )
                    new_java_hash_map[new_hash] = java_hash
                # endfor
            else:
                new_java_hash_map[java_method_id] = java_hash_list[0]
        # endfor

        # 1. collect unique cs method id
        cs_id_map = defaultdict(list)
        for cs_hash in cs_change_history:
            cs_method_id = cs_hash.split("/")[-1].replace(".cs", "").lower()
            cs_id_map[cs_method_id].append(cs_hash)
        # endfor
        new_cs_hash_map = {}
        for cs_method_id, cs_hash_list in cs_id_map.items():
            if len(cs_hash_list) > 1:
                smallest_index = -1
                for index in range(1, 20):
                    dir_index = -1 * index
                    cs_id_set = set()
                    for cs_hash in cs_hash_list:
                        new_hash = (
                            ".".join(cs_hash.split("/")[dir_index:])
                            .replace(".cs", "")
                            .lower()
                        )
                        cs_id_set.add(new_hash)
                    # endfor
                    if len(cs_id_set) == len(cs_hash_list):
                        smallest_index = dir_index
                        break
                for cs_hash in cs_hash_list:
                    new_hash = (
                        ".".join(cs_hash.split("/")[smallest_index:])
                        .replace(".cs", "")
                        .lower()
                    )
                    new_cs_hash_map[new_hash] = cs_hash
                # endfor
            else:
                new_cs_hash_map[cs_method_id] = cs_hash_list[0]
        # endfor
        return new_java_hash_map, new_cs_hash_map

    # 1.4.1
    def align_project_java_csharp_method(self):
        """Align the java and csharp method for all the projects."""
        for java_project in projects_map:
            self.align_java_csharp_method(java_project)

    # 1.4.2 Build dict: method_hash -> {java: [method diffs], c#: [method diffs]}=
    def collect_method_diff_history(self):
        """Collect aligned method diff history for all the projects"""
        for java_project in projects_map:
            self.collect_project_diff_history(java_project)

    # 1.4.3 filter the aligned diff
    def filter_aligned_diff(self, java_first: bool = True):
        """Filter the aligned diff for all the projects"""
        for java_project in projects_map:
            self.filter_project_aligned_diff(java_project, java_first=java_first)

    # 2
    def build_delta_translation_dataset(self, filter_type: str = "time+sim"):
        """Build the dataset: project, java-SHA, java-new, java-old, cs-SHA, cs-new, cs-old."""

        projects = projects_map.keys()
        dataset = []
        for project in projects:
            cs_project = projects_map[project]
            self.logger.info(f"Building dataset for {project}")
            no_diff_count = 0
            file_not_found_error = 0
            missing_cs_method_count = 0

            aligned_diff_data = io.load(
                self.results_dir / f"{filter_type}-{project}-aligned-method-diff.json"
            )
            for method_info in tqdm(aligned_diff_data, total=len(aligned_diff_data)):
                java_method_info = method_info["java"]
                # 1. extract Java code
                java_old_method, java_new_method = None, None
                old_sha, new_sha = (
                    java_method_info["old-sha"],
                    java_method_info["new-sha"],
                )

                # 1.1 get old_method
                try:
                    old_sha_methods = io.load(
                        self.repos_results_dir
                        / project
                        / "collector"
                        / f"java-method-data-{old_sha}.json"
                    )
                except FileNotFoundError:
                    self.logger.error(
                        f"Cannot find old SHA {old_sha} for java project {project}"
                    )
                    file_not_found_error += 1
                    continue

                java_del_code = [code[2:] for code in java_method_info["del-code"]]
                java_add_code = [code[2:] for code in java_method_info["add-code"]]

                for j_m in old_sha_methods:
                    bad_code = False
                    if (
                        f"{j_m['path']}.{j_m['class_name']}.{j_m['name']}-{j_m['params']}"
                        == java_method_info["old-method-hash"]
                    ):
                        # to make sure the data is correct
                        for del_code in java_del_code:
                            if " ".join(del_code.split()) not in " ".join(
                                j_m["code"].split()
                            ):
                                bad_code = True
                                break
                        if bad_code:
                            continue
                        java_old_method = j_m
                        break
                # sanity check
                if java_old_method is None:
                    raise RuntimeError(
                        f"Cannot find old version of Java method: {java_method_info['new-method-hash']} on SHA {old_sha}."
                    )
                # 1.2 get new_method
                try:
                    new_sha_methods = io.load(
                        self.repos_results_dir
                        / project
                        / "collector"
                        / f"java-method-data-{new_sha}.json"
                    )
                except FileNotFoundError:
                    self.logger.error(
                        f"Cannot find new SHA {new_sha} for java project {project}"
                    )
                    file_not_found_error += 1
                    continue

                for j_m in new_sha_methods:
                    bad_code = False
                    if (
                        f"{j_m['path']}.{j_m['class_name']}.{j_m['name']}-{j_m['params']}"
                        == java_method_info["new-method-hash"]
                    ):
                        for add_code in java_add_code:
                            if " ".join(add_code.split()) not in " ".join(
                                j_m["code"].split()
                            ):
                                bad_code = True
                                break
                        if bad_code:
                            continue
                        java_new_method = j_m
                        break
                # sanity check
                if java_new_method is None:
                    raise RuntimeError(
                        f"Cannot find new version of Java method : {java_method_info['new-method-hash']} on SHA {new_sha}."
                    )
                assert java_old_method["code"] != java_new_method["code"]

                # 2. extract C# code
                cs_old_method, cs_new_method = None, None
                cs_method_info = method_info["cs"]
                old_sha, new_sha = (
                    cs_method_info["old-sha"],
                    cs_method_info["new-sha"],
                )

                # 2.1 get old_method
                try:
                    old_sha_methods = io.load(
                        self.repos_results_dir
                        / cs_project
                        / "collector"
                        / f"cs-method-data-{old_sha}.json"
                    )
                except FileNotFoundError:
                    self.logger.error(
                        f"Cannot find old SHA {old_sha} for C# project {cs_project}"
                    )
                    file_not_found_error + 1
                    continue

                cs_del_code = [
                    code[2:]
                    for code in cs_method_info["del-code"]
                    if not code[2:].startswith("//")
                ]
                cs_add_code = [
                    code[2:]
                    for code in cs_method_info["add-code"]
                    if not code[2:].startswith("//")
                ]

                for c_m in old_sha_methods:
                    bad_code = False
                    if (
                        f"{c_m['path']}.{c_m['class_name']}.{c_m['name']}-{c_m['params']}"
                        == cs_method_info["old-method-hash"]
                    ):
                        # to make sure the data is correct
                        for del_code in cs_del_code:
                            if "//" in c_m["code"]:
                                c_m["code"] = re.sub(
                                    line_comment_pattern, "", c_m["code"]
                                )
                            if "/*" in c_m["code"]:
                                c_m["code"] = re.sub(
                                    block_comment_pattern, "", c_m["code"]
                                )
                            if " ".join(del_code.split()) not in " ".join(
                                c_m["code"].split()
                            ):
                                bad_code = True
                                break
                        if bad_code:
                            continue
                        cs_old_method = c_m
                        break
                if cs_old_method is None:
                    # raise RuntimeError(
                    #     f"Cannot find old version of C# method: {cs_method_info['old-method-hash']} on SHA {old_sha}."
                    # )
                    self.logger.warning(
                        f"Cannot find old version of C# method: {cs_method_info['old-method-hash']} on SHA {old_sha}."
                    )
                # 2.2 get new_method
                try:
                    new_sha_methods = io.load(
                        self.repos_results_dir
                        / cs_project
                        / "collector"
                        / f"cs-method-data-{new_sha}.json"
                    )
                except FileNotFoundError:
                    self.logger.error(
                        f"Cannot find new SHA {new_sha} for C# project {cs_project}"
                    )
                    file_not_found_error += 1
                    continue
                for c_m in new_sha_methods:
                    bad_code = False
                    if (
                        f"{c_m['path']}.{c_m['class_name']}.{c_m['name']}-{c_m['params']}"
                        == cs_method_info["new-method-hash"]
                    ):
                        for add_code in cs_add_code:
                            if "//" in c_m["code"]:
                                c_m["code"] = re.sub(
                                    line_comment_pattern, "", c_m["code"]
                                )
                            if "/*" in c_m["code"]:
                                c_m["code"] = re.sub(
                                    block_comment_pattern, "", c_m["code"]
                                )
                            if " ".join(add_code.split()) not in " ".join(
                                c_m["code"].split()
                            ):
                                bad_code = True
                                break
                        if bad_code:
                            continue
                        cs_new_method = c_m
                        break
                if cs_new_method is None:
                    self.logger.warning(
                        f"Cannot find new version of c# method: {cs_method_info['new-method-hash']} on SHA {new_sha}."
                    )
                if cs_new_method is None or cs_old_method is None:
                    missing_cs_method_count += 1
                    continue
                assert cs_old_method["code"] != cs_new_method["code"]

                if (
                    java_old_method["code"].split() == java_new_method["code"].split()
                    or cs_old_method["code"].split() == cs_new_method["code"].split()
                ):
                    no_diff_count += 1
                    continue
                dataset.append(
                    {
                        "project": project,
                        "java-SHA": java_method_info["new-sha"],
                        "java-old": java_old_method,
                        "java-new": java_new_method,
                        "java-commit-date": java_method_info["commit-date"],
                        "cs-SHA": cs_method_info["new-sha"],
                        "cs-old": cs_old_method,
                        "cs-new": cs_new_method,
                        "cs-commit-date": cs_method_info["commit-date"],
                    }
                )
            self.logger.info(f"Total no diff data is {no_diff_count} pairs.")
            self.logger.info(
                f"Total file not found error is {file_not_found_error} pairs, and missing method is {missing_cs_method_count} for project {project}."
            )

        self.logger.info(f"In total collect {len(dataset)} pairs.")
        io.dump(
            Macros.raw_data_dir / "delta-translation-dataset-cs2java.jsonl",
            dataset,
            io.Fmt.jsonList,
        )

    # 3. split based on commit date
    def time_sort_dataset(self, data_file: str, target_lang: str = "cs"):
        """Sort the dataset by the date of the commit."""

        data_list = io.load(Macros.raw_data_dir / data_file)
        time_sorted_data = []

        current_project = ""
        project_data = []
        for dt in tqdm(data_list, total=len(data_list)):
            prj = dt["project"]

            if prj != current_project and current_project != "":
                # process all data in the prior project
                sorted_project_data = sorted(
                    project_data, key=lambda x: x[f"{target_lang}-commit-date"]
                )  # sorted from old to new
                self.logger.info(
                    f"project {current_project} has {len(sorted_project_data)} data points."
                )
                time_sorted_data.extend(sorted_project_data)
                project_data = [dt]
                current_project = prj
            elif current_project == "":
                project_data = [dt]
                current_project = prj
            else:
                project_data.append(dt)
        sorted_project_data = sorted(
            project_data, key=lambda x: x[f"{target_lang}-commit-date"]
        )  # sorted from old to new
        self.logger.info(
            f"project {current_project} has {len(sorted_project_data)} data points."
        )
        time_sorted_data.extend(sorted_project_data)
        io.dump(
            Macros.raw_data_dir / "delta-translation-dataset-time-sorted.jsonl",
            time_sorted_data,
            io.Fmt.jsonList,
        )

    # 4. time segement dataset
    def time_segment_dataset(self):
        """Split the raw data into by time based on Java project."""

        data_list = io.load(
            Macros.raw_data_dir / "delta-translation-dataset-time-sorted.jsonl"
        )

        split_date = {}  # the date to split train, valid, test set
        train_set = []
        valid_set = []
        test_set = []

        project_data = defaultdict(list)

        for dt in data_list:
            project_data[dt["project"]].append(dt)

        for prj, prj_data in project_data.items():
            total_size = len(prj_data)
            self.logger.info(f"project {prj} has {total_size} data points.")
            if len(prj_data) == 2:
                assert (
                    prj_data[0]["java-commit-date"] <= prj_data[1]["java-commit-date"]
                )
                if prj_data[0]["java-commit-date"] < prj_data[1]["java-commit-date"]:
                    train_set = 1
                    test_set = 1
                else:
                    test_set = 2
            else:
                train_size = int(total_size * (self.val_ratio + self.train_ratio))
                test_size = total_size - train_size
                if train_size == total_size:
                    test_size = 1
                while (
                    prj_data[train_size - 1]["cs-commit-date"]
                    >= prj_data[train_size]["cs-commit-date"]
                    and train_size > 0
                ):
                    train_size -= 1
                    test_size += 1
            # endif
            val_size = int(
                train_size * (self.val_ratio / (self.val_ratio + self.train_ratio))
            )
            if val_size == 0 and train_size > 1:
                val_size = 1
            train_size = train_size - val_size
            train_set.extend(prj_data[:train_size])
            valid_set.extend(prj_data[train_size : train_size + val_size])
            test_set.extend(prj_data[train_size + val_size :])

            # write down date
            split_date[prj] = {
                "train": {
                    "java": (
                        prj_data[0]["java-commit-date"],
                        prj_data[train_size - 1]["java-commit-date"],
                    ),
                    "cs": (
                        prj_data[0]["cs-commit-date"],
                        prj_data[train_size - 1]["cs-commit-date"],
                    ),
                },
                "valid": {
                    "java": (
                        prj_data[train_size]["java-commit-date"],
                        prj_data[train_size + val_size - 1]["java-commit-date"],
                    ),
                    "cs": (
                        prj_data[train_size]["cs-commit-date"],
                        prj_data[train_size + val_size - 1]["cs-commit-date"],
                    ),
                },
                "test": {
                    "java": (
                        prj_data[train_size + val_size]["java-commit-date"],
                        prj_data[-1]["java-commit-date"],
                    ),
                    "cs": (
                        prj_data[train_size + val_size]["cs-commit-date"],
                        prj_data[-1]["cs-commit-date"],
                    ),
                },
            }
            assert split_date[prj]["valid"]["cs"][1] < split_date[prj]["test"]["cs"][0]

        self.logger.info(
            f"{len(train_set)} training data, {len(valid_set)} validation data and {len(test_set)} test data."
        )
        io.dump(
            Macros.data_dir / "raw" / "delta-translation-train.jsonl",
            train_set,
            io.Fmt.jsonList,
        )
        io.dump(
            Macros.data_dir / "raw" / "delta-translation-valid.jsonl",
            valid_set,
            io.Fmt.jsonList,
        )
        io.dump(
            Macros.data_dir / "raw" / "delta-translation-test.jsonl",
            test_set,
            io.Fmt.jsonList,
        )

        io.dump(
            Macros.results_dir / "stats" / "stats-data-split-date.json",
            split_date,
            io.Fmt.jsonPretty,
        )

    # 4.1 projects segement dataset
    def projects_segment_dataset(self):
        """Split the raw data by projects based on Java project."""

        # setup
        raw_data_dir = Macros.data_dir / "raw"
        output_dir = raw_data_dir / "cross-project"
        io.mkdir(output_dir)
        data_list = io.load(
            raw_data_dir / "delta-translation-dataset-time-sorted.jsonl"
        )

        # because the projects data amount is not even, manually specify the split
        train_projects = ["itext_itext7"]
        valid_projects = ["terabyte_jgit"]
        test_projects = set()

        train_set = []
        valid_set = []
        test_set = []

        for dt in data_list:
            if dt["project"] in train_projects:
                train_set.append(dt)
            elif dt["project"] in valid_projects:
                valid_set.append(dt)
            else:
                test_set.append(dt)
                test_projects.add(dt["project"])

        self.logger.info(
            f"{len(train_set)} training data, {len(valid_set)} validation data and {len(test_set)} test data."
        )
        io.dump(
            output_dir / "delta-translation-train.jsonl",
            train_set,
            io.Fmt.jsonList,
        )
        io.dump(
            output_dir / "delta-translation-valid.jsonl",
            valid_set,
            io.Fmt.jsonList,
        )
        io.dump(
            output_dir / "delta-translation-test.jsonl",
            test_set,
            io.Fmt.jsonList,
        )
        io.dump(
            output_dir / "projects-split.json",
            {
                "training-projects": train_projects,
                "valida-projects": valid_projects,
                "test-projects": list(test_projects),
            },
        )

    def tokenize_collected_data(self, file_path: str):
        """Tokenize the collected data."""
        from deltr.exe.CodeTokenizer import CodeTokenizer
        import atexit

        tokenizer = CodeTokenizer(main_class="org.csevo.Tokenizer")
        tokenizer.setup()
        atexit.register(tokenizer.teardown)

        data_list = io.load(
            file_path,
            io.Fmt.jsonList,
        )
        new_data = []
        for dt in tqdm(data_list, total=len(data_list)):
            # java
            dt["java-new"]["tokenized_code"] = tokenizer.tokenize(
                dt["java-new"]["code"], "java"
            ).strip()
            dt["java-old"]["tokenized_code"] = tokenizer.tokenize(
                dt["java-old"]["code"], "java"
            ).strip()
            # c#
            dt["cs-new"]["tokenized_code"] = tokenizer.tokenize(
                dt["cs-new"]["code"], "cs"
            ).strip()
            dt["cs-old"]["tokenized_code"] = tokenizer.tokenize(
                dt["cs-old"]["code"], "cs"
            ).strip()

            # sanity check
            assert dt["cs-new"]["tokenized_code"] != dt["cs-old"]["tokenized_code"]
            new_data.append(dt)

        io.dump(
            file_path,
            new_data,
            io.Fmt.jsonList,
        )

    # Helper functions
    def collect_changed_methods_for_commit(
        self,
        project_name: str,
        lang: str,
        prev_sha: str,
        cur_sha: str,
    ):
        """Collect changed methods from changed files between two consecutive commits."""

        import json

        repo_results_dir = self.repos_results_dir / project_name / "collector"
        try:
            prev_sha_methods = io.load(
                repo_results_dir / f"{lang}-method-data-{prev_sha}.json"
            )
            cur_sha_methods = io.load(
                repo_results_dir / f"{lang}-method-data-{cur_sha}.json"
            )
        except json.decoder.JSONDecodeError:
            return []
        except:
            self.logger.warning(
                f"{cur_sha} and {prev_sha} can not be parsed for project {project_name}."
            )
            return []
        changed_methods_list = []

        io.dump(
            repo_results_dir / f"{lang}-method-data-{cur_sha}.json",
            cur_sha_methods,
            io.Fmt.jsonNoSort,
        )
        io.dump(
            repo_results_dir / f"{lang}-method-data-{prev_sha}.json",
            prev_sha_methods,
            io.Fmt.jsonNoSort,
        )
        for p_m in prev_sha_methods:
            for c_m in cur_sha_methods:
                new_path = c_m["path"]
                old_path = p_m["path"]  # we consider the exact match of two paths
                if new_path != old_path:
                    continue
                if (
                    p_m["name"] == c_m["name"]
                    and p_m["class_name"] == c_m["class_name"]
                    and p_m["params"] == c_m["params"]
                    and p_m["code"].split() != c_m["code"].split()
                ):
                    changed_methods_list.append((p_m, c_m))
                    break
        # end for
        return changed_methods_list

    def collect_methods_for_commit(
        self, project_name: str, lang: str, sha: str, changed_files: List[str]
    ):
        """Collect methods for a given SHA and given project."""

        downloads_dir = Macros.repos_downloads_dir / project_name
        # download repo
        project_url = f"git@github.com:{project_name.split('_')[0]}/{project_name.split('_')[1]}.git"
        if not downloads_dir.exists():
            self.logger.info(f"Cloning repo {project_name} ... ")
            with IOUtils.cd(self.repos_downloads_dir):
                try:
                    with TimeUtils.time_limit(300):
                        BashUtils.run(
                            f"git clone {project_url} {project_name}",
                            expected_return_code=0,
                        )
                    # end with
                except TimeoutException:
                    self.logger.info(
                        f"{project_name} exceeds time limit, ignore this one."
                    )
                    return
                except:
                    self.logger.warning(
                        f"Project {project_name} failed: {sys.exc_info()}"
                    )
                    return
            # end with
        with io.cd(downloads_dir):
            self.logger.info(f"Checkout {sha}")
            bash.run(f"git checkout {sha} -f", check_returncode=0)
            self.collect_method_data(
                project_url="",
                project_name=project_name,
                project_sha=sha,
                lang=lang,
                changed_files=changed_files,
            )

    def collect_method_data(
        self,
        project_url: str,
        project_name: str,
        project_sha: str = None,
        lang: str = "java",
        changed_files: List[str] = None,
    ):
        """Collect methods in the project. If changed_files are given, only collect methods in the files."""

        Environment.require_collector()

        # 0. Download repo
        downloads_dir = self.repos_downloads_dir / project_name
        results_dir = self.repos_results_dir / project_name

        IOUtils.mk_dir(results_dir)
        assert downloads_dir.exists()

        # 2. Use parser to parse project
        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.url = project_url

        # Get revision (SHA)
        with IOUtils.cd(downloads_dir):
            try:
                if project_sha:
                    BashUtils.run(
                        f"git checkout {project_sha} -f", expected_return_code=0
                    )
                else:
                    project_sha = bash.run(f"git rev-parse HEAD").stdout.strip()
            except:
                self.logger.warning(f"Project {project_name} failed: {sys.exc_info()}")
                return
            project_data.revision = project_sha

        project_data_file = results_dir / "project.json"
        IOUtils.dump(
            project_data_file, IOUtils.jsonfy(project_data), IOUtils.Format.jsonPretty
        )

        # Prepare config
        log_file = results_dir / "collector-log.txt"
        output_dir = results_dir / "collector"

        config = {
            "collect": True,
            "projectDir": str(downloads_dir),
            "projectDataFile": str(project_data_file),
            "logFile": str(log_file),
            "outputDir": str(output_dir),
            "lang": lang,
            "revision": str(project_data.revision),
        }
        if changed_files:
            config["fileNames"] = changed_files
        # self.logger.info(f"Project parsing config file: \n {config}")
        config_file = results_dir / "collector-config.json"
        IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

        self.logger.info(
            f"Starting the collector. Check log at {log_file} and outputs at {output_dir}"
        )
        rr = BashUtils.run(
            f"java -jar {Environment.collector_jar} {config_file}",
        )
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
        # end if

        return

    def collect_pymethod_data(
        self,
        project_url: str,
        project_name: str,
        project_sha: str = None,
        lang: str = "python",
        changed_files: List[str] = None,
    ):
        """Collect python methods in the project. If changed_files are given, only collect methods in the files."""

        # Environment.require_collector()

        # 0. Download repo
        downloads_dir = self.repos_downloads_dir / project_name
        results_dir = self.repos_results_dir / project_name

        IOUtils.mk_dir(results_dir)

        # Clone the repo if not exists
        if not downloads_dir.exists():
            self.logger.info(f"Cloning repo {project_name} ... ")
            with IOUtils.cd(self.repos_downloads_dir):
                try:
                    with TimeUtils.time_limit(300):
                        BashUtils.run(
                            f"git clone {project_url} {project_name}",
                            expected_return_code=0,
                        )
                    # end with
                except TimeoutException:
                    self.logger.info(
                        f"{project_name} exceeds time limit, ignore this one."
                    )
                    return
                except:
                    self.logger.warning(
                        f"Project {project_name} failed: {sys.exc_info()}"
                    )
                    return
            # end with
        # end if

        # 2. Use parser to parse project
        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.url = project_url

        # Get revision (SHA)
        with IOUtils.cd(downloads_dir):
            try:
                if project_sha:
                    BashUtils.run(
                        f"git checkout {project_sha} -f", expected_return_code=0
                    )
                else:
                    project_sha = bash.run(f"git rev-parse HEAD").stdout.strip()
            except:
                self.logger.warning(f"Project {project_name} failed: {sys.exc_info()}")
                return
            project_data.revision = project_sha

        project_data_file = results_dir / "project.json"
        IOUtils.dump(
            project_data_file, IOUtils.jsonfy(project_data), IOUtils.Format.jsonPretty
        )

        # Run python parser to parse python files
        py_func_dict = self.parse_python_methods(
            project_dir=downloads_dir / "runtime" / "Python3"
        )
        print(f"In total collected {len(py_func_dict)} functions")
        io.dump(
            Macros.results_dir / "temp-test-py-funcs.json",
            py_func_dict,
            io.Fmt.jsonPretty,
        )

        return

    def parse_python_methods(self, project_dir: str):
        """Parse all python methods in a project given directory."""

        from deltr.collector.PythonParser import PythonParser
        import dataclasses

        parser = PythonParser()
        pyfiles = []
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if str(file).endswith(".py"):
                    pyfiles.append(os.path.join(root, file))

        assert len(pyfiles) > 0, "No python files found."
        function_dict = {}
        for pyfile in pyfiles:
            functions = parser.collect_functions(project_dir / pyfile)
            if functions is None:
                continue
            for m in functions:
                function_dict[m.path + m.className + m.name] = dataclasses.asdict(m)

        return function_dict

    def filter_project_aligned_diff(
        self,
        java_project_name: str,
        filter_type: str = "time+sim",
        java_first: bool = True,
    ):
        """Find the aligned diff from the aligned histories and filter based on jaccard similarity."""

        aligned_method_history = io.load(
            self.results_dir / f"{java_project_name}-method-diff-history.json"
        )
        aligned_diff_data = []
        for method_history in tqdm(
            aligned_method_history, total=len(aligned_method_history)
        ):
            if filter_type == "time":
                matched_diff_list = self.match_diff_based_on_time(
                    method_history, java_project_name, java_first=java_first
                )
                aligned_diff_data.append(matched_diff_list)
            elif filter_type == "sim":
                matched_diff_list = self.match_diff_based_on_jaccard(method_history)
                aligned_diff_data.extend(matched_diff_list)
            else:
                matched_diff_list = self.match_diff_based_on_time(
                    method_history, java_project_name, java_first=java_first
                )  # [{}]
                matched_diff_list = self.match_diff_based_on_jaccard(matched_diff_list)
                aligned_diff_data.extend(matched_diff_list)

        self.logger.info(
            f"Collect {len(aligned_diff_data)} aligned method diff for project {java_project_name}."
        )
        io.dump(
            self.results_dir
            / f"{filter_type}-{java_project_name}-aligned-method-diff.json",
            aligned_diff_data,
            io.Fmt.jsonNoSort,
        )

    def match_diff_based_on_jaccard(self, method_diff_history: dict):
        """Find the matched diff based on smilarity."""
        method_diff_list = []
        if isinstance(method_diff_history, dict):
            java_diffs = method_diff_history["java"]
            cs_diffs = method_diff_history["cs"]
        elif isinstance(method_diff_history, list):
            java_diffs = [dt["java"] for dt in method_diff_history]
            cs_diffs = [dt["cs"] for dt in method_diff_history]

        cs_best_match_sim = defaultdict(float)
        for i, java_diff in enumerate(java_diffs):
            best_sim = 0
            diff_pair = None
            cs_diff_list = cs_diffs[i]
            for cs_diff in cs_diff_list:
                if cs_diff["add-tokens"] == [] and cs_diff["del-tokens"] == []:
                    continue
                token_sim, line_sim = self.compare_diff_similarity(java_diff, cs_diff)
                if (
                    token_sim < self.token_sim_threshold
                    or line_sim < self.line_sim_threshold
                ):
                    continue

                if token_sim + line_sim >= best_sim:
                    best_sim = token_sim + line_sim
                    diff_pair = {
                        "java": java_diff,
                        "cs": cs_diff,
                    }
            if diff_pair:
                cs_diff = diff_pair["cs"]
                exist_best_sim = cs_best_match_sim[
                    cs_diff["new-sha"]
                    + cs_diff["old-sha"]
                    + str(cs_diff["add-code"])
                    + str(cs_diff["del-code"])
                    + cs_diff["new-method-hash"]
                    + cs_diff["commit-date"]
                ]
                if exist_best_sim > 0 and exist_best_sim < best_sim:
                    # de-duplicate
                    duplicate = True
                    for method_diff_pair in method_diff_list:
                        if (
                            method_diff_pair["cs"]["new-sha"] == cs_diff["new-sha"]
                            and method_diff_pair["cs"]["old-sha"] == cs_diff["old-sha"]
                            and str(method_diff_pair["cs"]["add-code"])
                            == str(cs_diff["add-code"])
                            and str(method_diff_pair["cs"]["del-code"])
                            == str(cs_diff["del-code"])
                            and method_diff_pair["cs"]["new-method-hash"]
                            == cs_diff["new-method-hash"]
                            and method_diff_pair["cs"]["commit-date"]
                            == cs_diff["commit-date"]
                        ):
                            duplicate = False
                            method_diff_list.remove(method_diff_pair)
                            break
                    assert duplicate == False
                    method_diff_list.append(diff_pair)
                    cs_best_match_sim[
                        cs_diff["new-sha"]
                        + cs_diff["old-sha"]
                        + str(cs_diff["add-code"])
                        + str(cs_diff["del-code"])
                        + cs_diff["new-method-hash"]
                        + cs_diff["commit-date"]
                    ] = best_sim
                elif exist_best_sim == 0:
                    cs_best_match_sim[
                        cs_diff["new-sha"]
                        + cs_diff["old-sha"]
                        + str(cs_diff["add-code"])
                        + str(cs_diff["del-code"])
                        + cs_diff["new-method-hash"]
                        + cs_diff["commit-date"]
                    ] = best_sim
                    method_diff_list.append(diff_pair)

        return method_diff_list

    def match_diff_based_on_time(
        self, method_diff_history: dict, java_project_name: str, java_first: bool = True
    ):
        """Find the matched diff based on commit date."""

        matched_diff_list = []
        which_older = 1 if java_first else -1
        for java_diff in method_diff_history["java"]:
            potential_match_cs_diffs = []
            if java_diff["add-tokens"] == [] and java_diff["del-tokens"] == []:
                continue
            java_commit_date = datetime.strptime(
                get_commit_date(java_diff["new-sha"], java_project_name),
                "%Y-%m-%d %H:%M:%S",
            )
            java_diff["commit-date"] = str(java_commit_date)
            # min_time = float("inf")
            cs_project_name = projects_map[java_project_name]
            for cs_diff in method_diff_history["cs"]:
                if cs_diff["add-tokens"] == [] and cs_diff["del-tokens"] == []:
                    continue
                cs_commit_date = datetime.strptime(
                    get_commit_date(cs_diff["new-sha"], cs_project_name),
                    "%Y-%m-%d %H:%M:%S",
                )
                cs_diff["commit-date"] = str(cs_commit_date)
                delta_days = (java_commit_date - cs_commit_date).days
                if (
                    abs(delta_days) < self.commit_date_threshold
                    and which_older * delta_days < 0
                ):
                    potential_match_cs_diffs.append(cs_diff)
            if len(potential_match_cs_diffs) > 0:
                matched_diff_list.append(
                    {"java": java_diff, "cs": potential_match_cs_diffs}
                )
        return matched_diff_list

    def collect_project_diff_history(self, java_project_name: str):
        """Align the history of methods in c# and java."""

        method_align_history = []
        cs_project_name = projects_map[java_project_name]
        java_method_history = io.load(
            Macros.repos_results_dir
            / java_project_name
            / f"java-method-change-history.json"
        )
        cs_method_history = io.load(
            Macros.repos_results_dir
            / cs_project_name
            / f"cs-method-change-history.json"
        )
        java_cs_method_map = io.load(
            Macros.results_dir
            / "repo-data"
            / f"{java_project_name}-java-csharp-method-map.json"
        )
        for j_method_hash, c_method_hash in tqdm(java_cs_method_map.items()):
            if isinstance(c_method_hash, list):
                c_method_hash = c_method_hash[0]
            # endif
            method_align_history.append(
                {
                    "java": java_method_history[j_method_hash],
                    "cs": cs_method_history[c_method_hash],
                }
            )
        # endfor
        self.logger.info(
            f"Found {len(method_align_history)} methods with diff history in project {java_project_name}."
        )
        io.dump(
            self.results_dir / f"{java_project_name}-method-diff-history.json",
            method_align_history,
            io.Fmt.jsonNoSort,
        )

    def align_java_csharp_method(self, java_project_name: str):
        """
        1.4.1
        Build java and csharp method mapping.

        NOTE: it involves manually checking the mapping.
        """

        cs_project_name = projects_map[java_project_name]
        java_id_map, cs_id_map = self.build_unique_method_id(
            java_changed_methods_file=Macros.repos_results_dir
            / java_project_name
            / "java-method-change-history.json",
            cs_changed_methods_file=Macros.repos_results_dir
            / cs_project_name
            / "cs-method-change-history.json",
        )
        java_2_cs_map = self.map_java_to_cs(java_id_map, cs_id_map)
        self.logger.info(
            f"Size of methods mapping for project {java_project_name} is {len(java_2_cs_map)}"
        )
        io.dump(
            Macros.results_dir
            / "repo-data"
            / f"{java_project_name}-java-csharp-method-map.json",
            java_2_cs_map,
            io.Fmt.jsonPretty,
        )

    def map_java_to_cs(self, java_id_map: dict, cs_id_map: dict):
        """Map java unique id with c# unique id."""

        java_2_cs_methods = {}
        uncertain_match_count = 0
        for java_id, java_hash in tqdm(java_id_map.items()):
            if java_id in cs_id_map:
                java_2_cs_methods[java_hash] = cs_id_map[java_id]
            else:
                # try manually inspect
                possible_match_cs_methods = []
                for cs_id, cs_hash in cs_id_map.items():
                    if java_id.split("-")[0] == cs_id.split("-")[0]:
                        # first check the number of parameters
                        if str(java_id.count("[")) != str(cs_id.count("[")):
                            continue
                        temp_cs_id = copy.deepcopy(cs_id)
                        for cs_type, java_type in cs_java_type_map.items():
                            temp_cs_id = temp_cs_id.replace(cs_type, java_type)
                        if java_id == temp_cs_id:
                            java_2_cs_methods[java_hash] = cs_hash
                            break
                        else:
                            possible_match_cs_methods.append(cs_id)
                # endfor

                # if len(possible_match_cs_methods) > 0:
                #     java_2_cs_methods[java_hash] = [
                #         cs_id_map[cs_id] for cs_id in possible_match_cs_methods
                #     ]
                #     uncertain_match_count += 1

                if len(possible_match_cs_methods) > 0:
                    max_sim = 0
                    match_cs_id = ""
                    for cs_id in possible_match_cs_methods:
                        param_sim = similar(java_id.split("-")[1], cs_id.split("-")[1])
                        if param_sim > max_sim and param_sim > 0.5:
                            max_sim = param_sim
                            match_cs_id = cs_id
                    if match_cs_id != "":
                        if max_sim < 0.8:
                            uncertain_match_count += 1
                            java_2_cs_methods[java_hash] = [cs_id_map[match_cs_id]]
                        else:
                            java_2_cs_methods[java_hash] = cs_id_map[match_cs_id]
        self.logger.info(f"Uncertain match count: {uncertain_match_count}")

        return java_2_cs_methods

    def compare_diff_similarity(
        self,
        java_diff: dict,
        cs_diff: dict,
    ) -> Tuple[float, float]:
        """Algorithm to find the aligned pair"""

        add_java_tks, del_java_tks = java_diff["add-tokens"], java_diff["del-tokens"]
        add_cs_tks, del_cs_tks = cs_diff["add-tokens"], cs_diff["del-tokens"]
        # 1. token level similarity
        if len(add_java_tks) == 0 and len(add_cs_tks) == 0:
            add_tokens_similarity = self.token_sim_threshold
        else:
            add_tokens_similarity = self.compute_diff_similarity(
                add_java_tks,
                add_cs_tks,
            )
        if len(del_java_tks) == 0 and len(del_cs_tks) == 0:
            del_tokens_similarity = self.token_sim_threshold
        else:
            del_tokens_similarity = self.compute_diff_similarity(
                del_java_tks, del_cs_tks
            )
        tokens_sim = add_tokens_similarity * (0.5) + del_tokens_similarity * (0.5)
        # 2. line level similarity
        add_line_similarity = self.compute_diff_similarity(
            java_diff["add-code"], cs_diff["add-code"]
        )
        del_line_similarity = self.compute_diff_similarity(
            java_diff["del-code"], cs_diff["del-code"]
        )
        line_sim = add_line_similarity * 0.5 + del_line_similarity * 0.5

        return tokens_sim, line_sim

    def compute_diff_similarity(
        self, sha1_lines: List[str], sha2_lines: List[str], task: str = None
    ):
        """Compute similarity between the changes (added lines & deleted lines)"""

        sha1_diff = tokenize_code(" ".join(sha1_lines).strip().lower())
        sha2_diff = tokenize_code(" ".join(sha2_lines).strip().lower())
        if task == "inclusion":
            jaccard_sim = include_jaccard(sha1_diff, sha2_diff)
        else:
            jaccard_sim = jaccard(sha1_diff, sha2_diff)

        return jaccard_sim

    def remove_csharp_comment(self, cs_code: str):
        line_comment_pattern = r"//(.*?)\n"
        block_comment_pattern = r"/\*(.*?)\*/"
        cs_code = re.sub(line_comment_pattern, "", cs_code)
        cs_code = re.sub(block_comment_pattern, "", cs_code)

        return cs_code

    # temp script
    def sample_filtered_diff(self, filter_type: str = "time+sim"):
        """Sample the filtered method diff for inspection."""

        total_number = 0
        sample_diff_method_list = []
        for java_project in projects_map:
            filtered_aligned_diff = io.load(
                self.results_dir
                / f"{filter_type}-{java_project}-aligned-method-diff.json"
            )
            total_number += len(filtered_aligned_diff)
            K = 20
            sample_ids = random.choices(range(len(filtered_aligned_diff)), k=K)
            for i, dt in enumerate(filtered_aligned_diff):
                if i in sample_ids:
                    del dt["cs"]["add-tokens"]
                    del dt["cs"]["del-tokens"]
                    del dt["java"]["add-tokens"]
                    del dt["java"]["del-tokens"]
                    sample_diff_method_list.append(dt)
        self.logger.info(f"In total collect {total_number} pairs of changes.")
        io.dump(
            Macros.raw_data_dir / f"{filter_type}-filtered-sampled-aligned-diff.json",
            sample_diff_method_list,
            io.Fmt.jsonPretty,
        )

    def analyze_project_aligned_diff(self, java_project_name: str):
        """
        This method is designed for finding the aligned diff from the aligned histories and
        filter based on jaccard similarity and inspect the examples.
        """

        aligned_method_history = io.load(
            self.results_dir / f"{java_project_name}-method-diff-history.json"
        )
        within_time_not_sim = []
        no_diff_within_time = []
        for method_history in tqdm(
            aligned_method_history, total=len(aligned_method_history)
        ):
            time_matched_diff_list = self.match_diff_based_on_time(
                method_history, java_project_name
            )
            if len(time_matched_diff_list) == 0:
                for mh in method_history["java"]:
                    del mh["add-tokens"]
                    del mh["del-tokens"]
                for mh in method_history["cs"]:
                    del mh["add-tokens"]
                    del mh["del-tokens"]
                no_diff_within_time.append(method_history)
            sim_matched_diff_list = self.match_diff_based_on_jaccard(
                time_matched_diff_list
            )
            for dt in time_matched_diff_list:
                if dt not in sim_matched_diff_list:
                    try:
                        del dt["java"]["add-tokens"]
                        del dt["java"]["del-tokens"]
                        del dt["cs"]["add-tokens"]
                        del dt["cs"]["del-tokens"]
                    except:
                        pass
                    within_time_not_sim.append(dt)

        io.dump(
            self.results_dir
            / f"time-not-sim-{java_project_name}-aligned-method-diff.json",
            within_time_not_sim,
            io.Fmt.jsonNoSort,
        )
        io.dump(
            self.results_dir
            / f"not-time-not-sim-{java_project_name}-aligned-method-diff.json",
            no_diff_within_time,
            io.Fmt.jsonNoSort,
        )


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(RealDiffCollector, as_positional=False)
