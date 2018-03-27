#!/bin/bash
# script to automatically generate the psyplot api documentation using
# sphinx-apidoc and sed
sphinx-apidoc -f -M -e  -T -o api ../iucm/ ../iucm/_dist.*
# replace chapter title in iucm.rst

if [[ `uname` == 'Darwin' ]]; then
    INPLACE='-i ""'
else
    INPLACE='-i'
fi

sed $INPLACE -e 1,1s/.*/'Python API Reference'/ api/iucm.rst
sed $INPLACE -e 2,2s/.*/'===================='/ api/iucm.rst

sed $INPLACE '/Python/ i\
.. _iucm-api-reference:\
\
' api/iucm.rst
