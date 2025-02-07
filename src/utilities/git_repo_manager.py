import os
from datetime import datetime
from loguru import logger


class GitRepoManager:
    def __init__(self, logdir, prefix=""):
        self.logdir = logdir
        self.prefix = prefix
        self.commit_number, self.branch_name, self.git_diff_info = (
            self.get_git_repo_status()
        )
        logger.info(
            f"Commit Number: {self.commit_number}, Branch Name: {self.branch_name}"
        )
        self.setup_directory()

    def get_git_repo_status(self):
        commit_number = os.popen('git log -1 --pretty=format:"%h"').read().strip()
        branch_name = os.popen("git branch --show-current").read().strip()
        git_diff_info = os.popen("git diff").read()
        return commit_number, branch_name, git_diff_info

    def setup_directory(self):
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        name_components = [
            self.prefix,
            time_stamp,
            self.branch_name,
            self.commit_number,
        ]
        # Filter out empty strings to avoid unnecessary underscores
        directory_name = "_".join(filter(None, name_components))
        self.full_logdir = os.path.join(self.logdir, directory_name)
        if not os.path.exists(self.full_logdir):
            os.makedirs(self.full_logdir)
        self.save_git_diff()
        logger.info(f"save git diff info to {self.full_logdir}")

    def save_git_diff(self):
        diff_path = os.path.join(self.full_logdir, "diff.txt")
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(self.git_diff_info)
