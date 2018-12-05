#!/usr/bin/env python
import os
import conda_build.api
import subprocess as spr
fnames = list(conda_build.api.get_output_file_paths('conda-recipe'))
if os.getenv("TRAVIS") == "true":
    branch = os.getenv("TRAVIS_BRANCH")
else:
    branch = os.getenv("APPVEYOR_REPO_BRANCH")
spr.check_call(
    ['anaconda', '-t', os.getenv('CONDA_REPO_TOKEN'), 'upload', '-l', branch,
     '--force'] + fnames
)
