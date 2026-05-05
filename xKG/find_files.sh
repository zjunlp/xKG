#!/bin/bash
for keyword in "$@"; do
    echo "=== $keyword (in filename) ==="
    find . -type f -name "*$keyword*" -print
done
