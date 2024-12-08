import random
from collections import defaultdict
import re
from typing import *
from pathlib import Path
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
import math

from deltr.Environment import Environment
from deltr.collector.ProjectData import ProjectData
from deltr.Macros import Macros


projects_map = {
    "antlr_antlr4": "tunnelvisionlabs_antlr4cs",
    "apache_lucene": "apache_lucenenet",
    "apache_poi": "nissl-lab_npoi",
    "itext_itext7": "itext_itext7-dotnet",
    "formicary_fpml-toolkit-java": "formicary_fpml-toolkit-csharp",
    "eclipse_jgit": "mono_ngit",
    "quartz-scheduler_quartz": "quartznet_quartznet",
    "terabyte_jgit": "mono_ngit",
    # "locationtech_jts": "NetTopologySuite_NetTopologySuite",
}

cs_port_date = {
    "tunnelvisionlabs_antlr4cs": "Feb 16, 2013",
    "apache_lucenenet": "Nov 21, 2005",
    "nissl-lab_npoi": "May 8, 2011",
    "mono_ngit": "Oct 7, 2010",
    "itext_itext7-dotnet": "Apr 8, 2016",
    "NetTopologySuite_NetTopologySuite": "Jul 29, 2006",
    "formicary_fpml-toolkit-csharp": "July 11, 2006",
    "apache_logging-log4net": "Jan 24, 2004",
    "nhibernate_nhibernate-core": "Feb 18, 2003",
    "nant_nant": "Aug 11, 2001",
    "quartznet_quartznet": "Dec 8, 2006",
    "nHapiNET_nHapi": "Mar 8, 2014",
}

project_branch = {
    "antlr_antlr4": "master",
    "tunnelvisionlabs_antlr4cs": "master",
    "apache_lucene": "main",
    "apache_lucenenet": "master",
    "apache_poi": "trunk",
    "nissl-lab_npoi": "master",
    "eclipse_jgit": "master",
    "mono_ngit": "master",
    "itext_itext7": "develop",  # develop?
    "itext_itext7-dotnet": "develop",
    "locationtech_jts": "master",
    "NetTopologySuite_NetTopologySuite": "develop",
    "formicary_fpml-toolkit-java": "master",
    "formicary_fpml-toolkit-csharp": "master",
    "terabyte_jgit": "master",
    "apache_logging-log4j": "release-2.x",
    "apache_logging-log4net": "master",
    "nhibernate_nhibernate-core": "master",
    "hibernate_hibernate-orm": "main",
    "apache_ant": "master",
    "nant_nant": "master",
    "quartz-scheduler_quartz": "master",
    "quartznet_quartznet": "main",
    "hapifhir_hapi-hl7v2": "master",
    "nHapiNET_nHapi": "master",
}


class DataCollector:
    logger = LoggingUtils.get_logger(
        __name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO
    )

    def __init__(self):
        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir
        self.repos_dir: Path = Macros.project_dir / "repos"
        self.results_dir = Macros.results_dir / "repo-data"
        self.raw_data_dir = Macros.data_dir / "raw"
        self.collected_projects_list = []
        self.collected_projects_file = (
            Macros.results_dir / "collected-github-java-repos.json"
        )
        if self.collected_projects_file.exists():
            self.collected_projects_list = IOUtils.load(self.collected_projects_file)
        else:
            self.collected_projects_list = []
        io.mkdir(self.repos_downloads_dir)
        io.mkdir(self.repos_results_dir)
        self.token_sim_threshold = 0.4
        self.overlap_token_sim_threshold = 0.6
        self.line_sim_threshold = 0.5

        return

    def collect_java_method_diff_for_data(self):
        """Collect java historical method changes for dataset augmentation."""
        java_projects = projects_map.keys()
        total_collected_data = []

        for java_prj in tqdm(java_projects, total=len(java_projects)):
            java_diffs = self.collect_java_changed_history(java_prj)
            total_collected_data.extend(java_diffs)

        io.dump(
            Macros.data_dir / "raw" / "java-only-history-diff.jsonl",
            total_collected_data,
            io.Fmt.jsonList,
        )

    def collect_java_project_data(self):
        """Collect Java project methods data."""

        java_projects_file = self.repos_dir / "java-projects.json"
        java_projects = io.load(java_projects_file)

        for prj_name, prj_url in java_projects.items():
            if prj_name in self.collected_projects_list:
                continue
            self.logger.info(f"Start to collect methods from project {prj_name}")
            self.collect_method_data(
                prj_url, prj_name
            )  # change the code if SHA is specified
            self.collected_projects_list.append(prj_name)
            # end for

        io.dump(self.collected_projects_file, self.collected_projects_list)

    def collect_csharp_project_data(self):
        """Collect C# project methods data."""
        csharp_projects_file = self.repos_dir / "csharp-projects.json"
        csharp_projects = io.load(csharp_projects_file)

        for prj_name, prj_url in csharp_projects.items():
            if prj_name in self.collected_projects_list:
                continue
            self.logger.info(f"Start to collect methods from project {prj_name}")
            self.collect_method_data(
                prj_url, prj_name, lang="csharp"
            )  # change the code if SHA is specified
            self.collected_projects_list.append(prj_name)
            # end for

        io.dump(self.collected_projects_file, self.collected_projects_list)

    def build_translation_augment_data(self):
        """Aggregate translation data for augmentation."""

        method_map = io.load(
            Macros.results_dir / "stats" / "stats-method-mapping-augment.json"
        )
        augment_data = []

        for j_prj in method_map:
            c_prj = projects_map[j_prj]
            java_methods = io.load(
                self.repos_results_dir / j_prj / "collector" / "java-method-data.json"
            )
            cs_methods = io.load(
                self.repos_results_dir / c_prj / "collector" / "csharp-method-data.json"
            )
            for j_id, c_id in method_map[j_prj]["map"].items():
                augment_data.append(
                    {
                        "project": j_prj,
                        "java-SHA": method_map[j_prj]["java-SHA"],
                        "java-old": "",
                        "java-new": java_methods[int(j_id)],
                        "cs-SHA": method_map[j_prj]["cs-SHA"],
                        "cs-old": "",
                        "cs-new": cs_methods[int(c_id)],
                    }
                )

        io.dump(
            Macros.data_dir / "raw" / "translation-augment-data.jsonl",
            augment_data,
            io.Fmt.jsonList,
        )

    def collect_project_commit_history(self):
        """Collect projects commit history with interested files."""

        for java_project, cs_project in projects_map.items():
            if cs_project == "antlr_antlr4-cs":
                cs_project = "antlr_antlr4"
            # 1. mine history from java project
            self.logger.info(f"Start to collect commit history of {java_project}")
            self.collect_project_history(java_project, lang="java")
            # 2. mine history from csharp project
            self.logger.info(f"Start to collect commit history of {cs_project}")
            self.collect_project_history(cs_project, lang="cs")
        # end for

    def check_miss_method(self, project_name: str, align_methods: dict):
        """Check miss method."""

        map = io.load(Macros.results_dir / f"{project_name}-method-hash-map.json")
        for align_method in align_methods:
            align_class_name = ".".join(align_method.split(".")[:-1])
            # print(align_class_name)
            if align_class_name not in map:
                self.logger.info(f"Miss method: {align_class_name}")

    def augment_translation_data(self):
        """Augment training data by collecting pure translation data."""

        augment_data_stats = {}
        for java_project in projects_map:
            augment_data_stats[
                java_project
            ] = self.build_java_csharp_method_map_for_project(java_project)
        io.dump(
            Macros.results_dir / "stats" / "stats-method-mapping-augment.json",
            augment_data_stats,
            io.Fmt.jsonPretty,
        )

    def build_java_csharp_method_map_for_project(self, java_project: str):
        """Build java and c# aligned methods map for a particular project."""

        dataset_test_date = io.load(
            Macros.results_dir / "stats" / "stats-data-split-date.json"
        )
        j_prj, c_prj = java_project, projects_map[java_project]

        # first find the correct commit
        java_test_date = dataset_test_date[java_project]["valid"]["java"]
        cs_test_date = dataset_test_date[java_project]["valid"]["cs"]
        earlier_date = java_test_date if java_test_date < cs_test_date else cs_test_date
        self.logger.info(f"Earlier date is {earlier_date} for {java_project}")
        # mine methods in projects
        with io.cd(Macros.repos_downloads_dir / java_project):
            bash.run("git checkout $(git branch --show-current)")
            java_sha = bash.run(
                f"git log --until='{earlier_date}' --first-parent --no-merges --pretty=format:'%H'"
            ).stdout.split("\n")[1][:8]
            self.logger.info(f"Mining java SHA {java_sha}")
            self.collect_method_data("", java_project, java_sha, "java")
        with io.cd(Macros.repos_downloads_dir / c_prj):
            bash.run("git checkout $(git branch --show-current)")
            cs_sha = bash.run(
                f"git log --until='{earlier_date}' --first-parent --no-merges --pretty=format:'%H'"
            ).stdout.split("\n")[1][:8]
            self.logger.info(f"Mining c# SHA {cs_sha}")
            self.collect_method_data("", c_prj, cs_sha, "cs")

        java_method_file = (
            self.repos_results_dir / j_prj / "collector" / f"java-method-data.json"
        )
        csharp_method_file = (
            self.repos_results_dir / c_prj / "collector" / f"csharp-method-data.json"
        )
        java_methods = io.load(java_method_file)
        csharp_methods = io.load(csharp_method_file)
        j2c_map = self.map_java_cs_methods(
            java_methods=java_methods, csharp_methods=csharp_methods
        )

        self.logger.info(f"Size of project {j_prj} map is {len(j2c_map)}.")

        results_dict = {
            "java-SHA": java_sha,
            "cs-SHA": cs_sha,
            "map": j2c_map,
        }

        return results_dict

    def map_java_cs_methods(self, java_methods: List[dict], csharp_methods: List[dict]):
        """Find java and c# method map, return the method id dict."""
        j2c_map = {}
        for j_m in tqdm(java_methods, total=(len(java_methods))):
            for c_m in csharp_methods:
                if (
                    c_m["name"].lower() == j_m["name"].lower()
                    and c_m["path"].split("/")[-1].split(".")[0].lower()
                    == j_m["path"].split("/")[-1].split(".")[0].lower()
                    and [p.lower() for p_list in c_m["params"] for p in p_list]
                    == [p.lower() for p_list in j_m["params"] for p in p_list]
                ):
                    if (
                        c_m["class_name"] is not None
                        and j_m["class_name"] is not None
                        and c_m["class_name"].lower() != j_m["class_name"].lower()
                    ):
                        continue
                    if str(j_m["id"]) not in j2c_map:
                        j2c_map[str(j_m["id"])] = str(c_m["id"])
                    else:
                        self.logger.info("Find duplicate mapping, ignore this one.")
                    break
                # end if
        return j2c_map

    def build_java_csharp_method_map(self):
        """Build java and c# aligned methods map."""

        method_hash_list = set()
        if (Macros.results_dir / "java-csharp-method-map.json").exists():
            j2c_prj_map = io.load(Macros.results_dir / "java-csharp-method-map.json")
        else:
            j2c_prj_map = {}
        # j2c_prj_map.pop("apache_poi-nissl-lab_npoi")
        for j_prj, c_prj in [("apache_logging-log4j", "apache_logging-log4net")]:
            duplicate_count = 0
            # if f"{j_prj}-{c_prj}" in j2c_prj_map:
            #     continue
            j2c_map = {}
            java_method_file = (
                self.repos_results_dir / j_prj / "collector" / "java-method-data.json"
            )
            csharp_method_file = (
                self.repos_results_dir / c_prj / "collector" / "csharp-method-data.json"
            )
            java_methods = io.load(java_method_file)
            csharp_methods = io.load(csharp_method_file)

            # rules: 1. same function names
            #        2. same parameters (types and names)
            #        3. same file name
            #        4. same class name
            for j_m in tqdm(java_methods, total=(len(java_methods))):
                for c_m in csharp_methods:
                    if (
                        c_m["name"].lower() == j_m["name"].lower()
                        and c_m["path"].split("/")[-1].split(".")[0].lower()
                        == j_m["path"].split("/")[-1].split(".")[0].lower()
                        # and [p.lower() for p_list in c_m["params"] for p in p_list]
                        # == [p.lower() for p_list in j_m["params"] for p in p_list]
                    ):
                        # if (
                        #     c_m["class_name"] is not None
                        #     and j_m["class_name"] is not None
                        #     and c_m["class_name"].lower() != j_m["class_name"].lower()
                        # ):
                        #     continue
                        if j_m["id"] not in j2c_map:
                            j2c_map[j_m["id"]] = str(c_m["id"])
                            java_method_hash = (
                                j_m["path"].split("/")[-1].replace(".java", "").lower()
                                + f".{j_m['class_name'].lower()}"
                                + f".{j_m['name'].lower()}"
                            )
                            cs_method_hash = (
                                c_m["path"].split("/")[-1].replace(".cs", "").lower()
                                + f".{c_m['class_name'].lower()}"
                            )

                            method_hash_list.add(cs_method_hash)
                        else:
                            self.logger.info("Find duplicate mapping, ignore this one.")
                            duplicate_count += 1
                        break
                    # end if

            self.logger.info(
                f"Size of project {j_prj} map is {len(j2c_map)}. Duplicate cases are ignored: {duplicate_count}"
            )
            j2c_prj_map[f"{j_prj}-{c_prj}"] = j2c_map
            io.dump(
                Macros.results_dir / f"{j_prj}-method-hash-map.json", method_hash_list
            )
        io.dump(Macros.results_dir / "java-csharp-method-map.json", j2c_prj_map)

    def build_java_csharp_file_map(self):
        """Build map between Java and C# files."""

        method_maps = io.load(Macros.results_dir / "java-csharp-method-map.json")
        if (Macros.results_dir / "java-csharp-file-map.json").exists():
            prj_file_maps = io.load(Macros.results_dir / "java-csharp-file-map.json")
        else:
            prj_file_maps = {}
        if (Macros.results_dir / "java-csharp-mapped-files.json").exists():
            prj_mapped_files = io.load(
                Macros.results_dir / "java-csharp-mapped-files.json"
            )
            prj_mapped_files = defaultdict(list, prj_mapped_files)
        else:
            prj_mapped_files = defaultdict(list)

        for j_prj, cs_prj in projects_map.items():
            if f"{j_prj}-{cs_prj}" in prj_file_maps:
                continue
            prj_file_maps[f"{j_prj}-{cs_prj}"] = {}
            method_id_map = method_maps[f"{j_prj}-{cs_prj}"]
            java_methods = io.load(
                self.repos_results_dir / j_prj / "collector" / "java-method-data.json"
            )
            csharp_methods = io.load(
                self.repos_results_dir
                / cs_prj
                / "collector"
                / "csharp-method-data.json"
            )
            for j_mid, c_mid in method_id_map.items():
                j_m_file = java_methods[int(j_mid)]["path"]
                c_m_file = csharp_methods[int(c_mid)]["path"]
                prj_file_maps[f"{j_prj}-{cs_prj}"][j_m_file] = c_m_file
                prj_mapped_files[j_prj].append(j_m_file)
                prj_mapped_files[cs_prj].append(c_m_file)
            # remove duplicate file name
            prj_mapped_files[j_prj] = list(set(prj_mapped_files[j_prj]))
            prj_mapped_files[cs_prj] = list(set(prj_mapped_files[cs_prj]))
        # end for
        io.dump(Macros.results_dir / "java-csharp-file-map.json", prj_file_maps)
        io.dump(Macros.results_dir / "java-csharp-mapped-files.json", prj_mapped_files)

    def remove_comments_from_csharp(self):
        """Remove comments from csharp code."""

        line_comment_pattern = r"//(.*?)\n"
        block_comment_pattern = r"/\*(.*?)\*/"
        for prj in projects_map.values():
            csharp_method_file = (
                self.repos_results_dir / prj / "collector" / "csharp-method-data.json"
            )
            csharp_methods = io.load(csharp_method_file)
            for m in csharp_methods:
                if "//" in m["code"]:
                    m["code"] = re.sub(line_comment_pattern, "", m["code"])
                if "/*" in m["code"]:
                    m["code"] = re.sub(block_comment_pattern, "", m["code"])
        # end for
        io.dump(csharp_method_file, csharp_methods, io.Fmt.jsonPretty)

    # -- Helper functions -----------------------------------------------------

    def sample_git_history(self):
        """Sample git history of methods for manually checking."""
        K = 5

        history_to_check = defaultdict(list)
        for j_prj in tqdm(projects_map, total=len(projects_map)):
            aligned_method_history = io.load(
                self.results_dir / f"{j_prj}-method-aligned-history.json"
            )
            total_size = len(aligned_method_history)
            sample_ids = random.choices(range(total_size), k=K)
            for i, dt in enumerate(aligned_method_history):
                if i in sample_ids:
                    history_to_check[j_prj].append((dt, aligned_method_history[dt]))
        io.dump(
            Macros.data_dir / "raw" / "manually-check-history.json",
            history_to_check,
            io.Fmt.jsonPretty,
        )

    def collect_commit_date(self):
        """Collect commit date for each data"""

        data_list = io.load(Macros.data_dir / "raw" / "delta-translation-dataset.jsonl")
        for dt in tqdm(data_list, total=len(data_list)):
            # add java commit date
            sha = dt["java-SHA"].split("-")[1]
            prj = dt["project"]
            branch_name = project_branch[prj]
            with io.cd(Macros.repos_downloads_dir / prj):
                bash.run(f"git checkout {branch_name} -f")
                commit_date = bash.run(
                    f"git show -s --format=%cd --date=format:'%Y-%m-%d %H:%M:%S' {sha}",
                    check_returncode=0,
                ).stdout.strip()
            dt["java-commit-date"] = commit_date

            # add csharp commit date
            sha = dt["cs-SHA"].split("-")[1]
            prj = projects_map[dt["project"]]
            branch_name = project_branch[prj]
            with io.cd(Macros.repos_downloads_dir / prj):
                bash.run(f"git checkout {branch_name} -f")
                commit_date = bash.run(
                    f"git show -s --format=%cd --date=format:'%Y-%m-%d %H:%M:%S' {sha}",
                    check_returncode=0,
                ).stdout.strip()
            dt["cs-commit-date"] = commit_date
        # end for
        io.dump(
            Macros.data_dir / "delta-translation-dataset-w-date.jsonl",
            data_list,
            io.Fmt.jsonList,
        )

    def compare_overlap_similarity(
        self,
        java_diff: dict,
        cs_diff: dict,
    ) -> Tuple[float, float]:
        """Algorim to find the aligned pair"""

        add_java_tks, del_java_tks = java_diff["add-tokens"], java_diff["del-tokens"]
        add_cs_tks, del_cs_tks = cs_diff["add-tokens"], cs_diff["del-tokens"]
        # 1. token level similarity
        if len(add_java_tks) == 0 and len(add_cs_tks) == 0:
            add_tokens_similarity = self.overlap_token_sim_threshold
        else:
            if len(add_java_tks) == 0 or len(add_cs_tks) == 0:
                add_tokens_similarity = 0.0
            else:
                add_tokens_similarity = self.compute_diff_similarity(
                    add_java_tks, add_cs_tks, task="inclusion"
                )
        if len(del_java_tks) == 0 and len(del_cs_tks) == 0:
            del_tokens_similarity = self.overlap_token_sim_threshold
        else:
            if len(del_java_tks) == 0 or len(del_cs_tks) == 0:
                del_tokens_similarity = 0.0
            else:
                del_tokens_similarity = self.compute_diff_similarity(
                    del_java_tks, del_cs_tks, task="inclusion"
                )
        tokens_sim = add_tokens_similarity * (0.5) + del_tokens_similarity * (0.5)

        # 2. line level similarity
        add_line_similarity = self.compute_diff_similarity(
            java_diff["add-code"], cs_diff["add-code"], task="inclusion"
        )
        del_line_similarity = self.compute_diff_similarity(
            java_diff["del-code"], cs_diff["del-code"], task="inclusion"
        )
        line_sim = add_line_similarity * 0.5 + del_line_similarity * 0.5

        return tokens_sim, line_sim

    def compare_diff_similarity(
        self,
        java_diff: dict,
        cs_diff: dict,
    ) -> Tuple[float, float]:
        """Algorim to find the aligned pair"""

        add_java_tks, del_java_tks = java_diff["add-tokens"], java_diff["del-tokens"]
        add_cs_tks, del_cs_tks = cs_diff["add-tokens"], cs_diff["del-tokens"]
        # 1. token level similarity
        if len(add_java_tks) == 0 and len(add_cs_tks) == 0:
            add_tokens_similarity = 0.4
        else:
            add_tokens_similarity = self.compute_diff_similarity(
                add_java_tks,
                add_cs_tks,
            )
        if len(del_java_tks) == 0 and len(del_cs_tks) == 0:
            del_tokens_similarity = 0.4
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

    def collect_project_history(self, project_name: str, lang: str):
        """Collect git history of projct and extract the changed files that have aligned methods.

        Args:
            project_name (str): project name
            lang (str): java or cs
        """

        downloads_dir = self.repos_downloads_dir / project_name
        if not downloads_dir.exists():
            self.logger.warning(f"Project {project_name} not found.")
        if project_name == "antlr_antlr4" and lang == "cs":
            project_name = "antlr_antlr4-cs"
        prj_mapped_files = io.load(
            Macros.results_dir / "java-csharp-mapped-files.json"
        )[project_name]

        with IOUtils.cd(downloads_dir):
            BashUtils.run("git checkout master")
            shalist = BashUtils.run(
                f"git log --color=never --since='May 15 2011' --first-parent --no-merges --pretty=format:'%H'"
            ).stdout.split("\n")
            shalist = [sha[:8] for sha in shalist]
        shalist = shalist[::-1]
        # 2. Check the changed files in each commit

        target_commits = OrderedDict()
        time_order_sha = []
        with IOUtils.cd(downloads_dir):
            for i in tqdm(range(len(shalist) - 1)):
                cur_sha, pre_sha = shalist[i + 1], shalist[i]
                changed_files = BashUtils.run(
                    f"git diff {pre_sha} {cur_sha} --name-only"
                ).stdout.split("\n")
                changed_files = [f for f in changed_files if f.split(".")[-1] == lang]
                if (
                    len(changed_files)
                    > 0
                    # and len(set(changed_files).intersection(set(prj_mapped_files))) > 0
                ):
                    target_commits[f"{pre_sha}-{cur_sha}"] = changed_files

        self.logger.info(
            f"Collect {len(target_commits)} commits with {lang} files changed."
        )
        io.dump(
            self.repos_results_dir / project_name / "git-history.json",
            target_commits,
            io.Fmt.jsonNoSort,
        )

    def check_bad_examples(self):

        K = 50
        examples_to_check = io.load(Macros.data_dir / f"manual-check-{K}-examples.json")
        bad_examples = []
        for dt in examples_to_check:
            added_java_tks, del_java_tks = dt["add-java-tks"], dt["del-java-tks"]
            added_cs_tks, del_cs_tks = dt["add-cs-tks"], dt["del-cs-tks"]
            if len(added_java_tks) == 0 and len(added_cs_tks) == 0:
                add_tokens_similarity = 0.4
            else:
                add_tokens_similarity = self.compute_diff_similarity(
                    added_java_tks, added_cs_tks
                )
            if len(del_java_tks) == 0 and len(del_cs_tks) == 0:
                del_tokens_similarity = 0.4
            else:
                del_tokens_similarity = self.compute_diff_similarity(
                    del_java_tks, del_cs_tks
                )
            tokens_sim = add_tokens_similarity * (0.5) + del_tokens_similarity * (0.5)
            dt["tokens-sim"] = tokens_sim

            if tokens_sim < 0.4:
                print(dt["id"])
        # self.logger.info(f"Number of bad examples are {len(bad_examples)}")
        # io.dump(
        #     Macros.data_dir / "bad-manual-check-examples.json",
        #     bad_examples,
        #     io.Fmt.jsonNoSort,
        # )

    def code_tokenizer(self, raw_code: str, lang: str) -> Tuple[str, str]:
        """Tokenize both Java and C# code in the dataset"""
        from deltr.exe.CodeTokenizer import CodeTokenizer
        import atexit

        self.tokenizer = CodeTokenizer(main_class="org.csevo.Tokenizer")
        self.tokenizer.setup()
        atexit.register(self.tokenizer.teardown)

        # c# comment pattern
        line_comment_pattern = "\s//(.*?)\n"
        block_comment_pattern = r"/\*(.*?)\*/"

        if "//" in raw_code or "/*" in raw_code:
            code_no_comment = re.sub(line_comment_pattern, "\n", raw_code)
            code_no_comment = re.sub(block_comment_pattern, "", code_no_comment)
        else:
            code_no_comment = raw_code

        code_tok = self.tokenizer.tokenize(code_no_comment, lang).strip()
        return code_no_comment, code_tok


def compute_minimal_code_diffs(old_tokens: List[str], new_tokens: List[str]):

    added_tokens = []
    del_tokens = []

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(
        None, old_tokens, new_tokens
    ).get_opcodes():
        if edit_type == "equal":
            continue
        elif edit_type == "replace":
            added_tokens.extend(new_tokens[n_start:n_end])
            del_tokens.extend(old_tokens[o_start:o_end])

        elif edit_type == "insert":
            added_tokens.extend(new_tokens[n_start:n_end])
        else:
            del_tokens.extend(old_tokens[o_start:o_end])

    return added_tokens, del_tokens


if __name__ == "__main__":
    LoggingUtils.setup(LoggingUtils.INFO, Macros.log_file)
    CLI(DataCollector, as_positional=False)
