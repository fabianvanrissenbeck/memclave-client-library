#!/bin/bash

if [ ! -f subkernels.tar ];
then
    echo "The subkernels.tar file is missing. Did you already compile all subkernels?"
    exit 1
fi

tar xf subkernels.tar
mv subkernels/* ./
rmdir subkernels
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build --target benchmarks
