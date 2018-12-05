#!/usr/bin/env python
import os
import re
import conda_build.api
import subprocess as spr
fnames = list(conda_build.api.get_output_file_paths('conda-recipe'))
py_patt = re.compile('py\d\d')
repl = 'py' + os.getenv('PYTHON_VERSION').replace('.', '')
fnames = [py_patt.sub(repl, f) for f in fnames]
if os.getenv("TRAVIS") == "true":
    branch = os.getenv("TRAVIS_BRANCH")
else:
    branch = os.getenv("APPVEYOR_REPO_BRANCH")
spr.check_call(
    ['anaconda', '-t', os.getenv('CONDA_REPO_TOKEN'), 'upload', '-l', branch,
     '--force'] + fnames
)
