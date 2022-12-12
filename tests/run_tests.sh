#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.
set -e

# list of test cases you want to run
tests=(
    # test_other_things_on_root_level.py
    dual/dual_test.py
    expression/expression_test.py
    expression/variable_test.py
    expression/function_test.py
    expression/ops_test.py
    expression/compose_test.py
)

# Must add the module source path because we use `import cs107_package` in
# our test suite.  This is necessary if you want to test in your local
# development environment without properly installing the package.
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python3 ${@} -m unittest"
fi

# run the tests
${driver} ${tests[@]}