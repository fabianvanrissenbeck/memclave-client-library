#!/bin/bash
set -euo pipefail

# basic PrIM benchmark suite of 16 benchmarks
RUN_PRIM="no"

# BFS and MLP benchmarks with multiple inputs
RUN_BFSMLP="no"

# Microbenchmarks except subkernel load times/key exchange/ready line
RUN_MICRO="no"

# Microbenchmark reporting subkernel load times/key exchange and ready line
RUN_SUBK="no"

# Output Directory of CSV Files
OUTDIR=$(realpath output/)

function print_help_exit {
  echo "Usage: ./run.sh ([ all | prim | mlpbfs | micro | subk ])?"
  echo "Run benchmarks within the memclave environment. Usually no command line option is required."
  echo "You may alter the number of benchmarks executed by passing one of the following parameters:"
  echo "all    - Run all benchmarks even those requiring a modified TL"
  echo "prim   - Run the full PrIM benchmark suite with default input sizes"
  echo "bfsmlp - Run the MLP and BFS benchmarks with different input sizes"
  echo "micro  - Run the MRAM throughput and sealed EM transfer benchmarks"
  echo "subk   - Run the benchmark reporting ready line, key exchange and subkernel load speeds"
  echo "         Requires a modified TL."
  exit 1
}

if [ "$#" == "0" ];
then
  RUN_PRIM="yes"
  RUN_BFSMLP="yes"
  RUN_MICRO="yes"
elif [ "$#" == "1" ];
then
  if [ "$1" == "all" ];
  then
    RUN_PRIM="yes"
    RUN_BFSMLP="yes"
    RUN_MICRO="yes" 
    RUN_SUBK="yes"
  elif [ "$1" == "prim" ];
  then
    RUN_PRIM="yes"
  elif [ "$1" == "bfsmlp" ];
  then
    RUN_BFSMLP="yes"
  elif [ "$1" == "micro" ];
  then
    RUN_MICRO="yes"
  elif [ "$1" == "subk" ];
  then
    RUN_SUBK="yes"
  else
    print_help_exit
  fi
else
  print_help_exit
fi

echo "=== Benchmark Configuration ==="
echo "RUN_PRIM: $RUN_PRIM"
echo "RUN_BFSMLP: $RUN_BFSMLP"
echo "RUN_MICRO: $RUN_MICRO"
echo "RUN_SUBK: $RUN_SUBK"

echo "Writing outputs to: $OUTDIR"
echo ""
mkdir -p $OUTDIR

BFS_DATA_URL=https://zenodo.org/records/18307126/files/bfs-data.tar.zst

if [ ! -f ./data/LiveJournal1 ];
then
	if [ ! -f ./bfs-data.tar.zst ];
	then
		echo "BFS Example Graphs are Missing. Downloading from Zenodo."
		curl -o bfs-data.tar.zst $BFS_DATA_URL
	fi

	tar xf ./bfs-data.tar.zst --directory ./data/
	exit 1;
fi

if [ "$RUN_PRIM" == "yes" ];
then
  echo "=== Running PrIM Benchmarks ==="
  ./run_prim.sh all
  mv ./build/prim_results.csv $OUTDIR/
  echo ""
fi

if [ "$RUN_BFSMLP" == "yes" ];
then
  echo "=== Running MLP Benchmarks ==="
  python3 run_mlp.py --mode memclave --cwd build
  mv mlp_results.csv $OUTDIR/
  echo ""

  echo "=== Running BFS Benchmarks ==="
  python3 run_bfs.py --mode memclave --cwd ./build --graph-prefix ../data
  mv bfs_results.csv $OUTDIR/
  echo ""
fi

if [ "$RUN_MICRO" == "yes" ];
then
  echo "=== Running MRAM Benchmark (This one takes ~30 minutes) ==="
  cd build
  ./mram > mram.csv
  cd ..
  mv build/mram.csv $OUTDIR/
  echo ""

  echo "=== Running Sealed EM Benchmark ==="
  cd build
  ./crypto-bench > crypto.csv
  cd ..
  mv build/crypto.csv $OUTDIR/
  echo ""
fi

if [ "$RUN_SUBK" == "yes" ];
then
  echo "=== Running Micro Benchmarks ==="
  cd build
  ./sk-load-bench > sk.csv
  cd ..
  mv build.sk.csv $OUTDIR/
  echo ""
fi
