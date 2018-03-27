#!/bin/bash
# script to automatically generate the psyplot api documentation using
# sphinx-apidoc and sed
sphinx-apidoc -f -M -e  -T -o api ../iucm/ ../iucm/_dist.*
# replace chapter title in iucm.rst

sed -i -e 1,1s/.*/'Python API Reference'/ api/iucm.rst
sed -i -e 2,2s/.*/'===================='/ api/iucm.rst

sed -i '/Python/ i\
.. _iucm-api-reference:\
\
' api/iucm.rst
