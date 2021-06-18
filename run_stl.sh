#!/bin/bash
cd stl_experiments/experiments
# repeat six times for different seeds
for i in {1..6};
do
    ./stl10_single.sh --augment --S 1 --deltaorth --model rbffdwrn16_8_stl --restrict 3
    ./stl10_single.sh --augment --S 1 --deltaorth --model rbffdwrn16_8_stl --N 4 --restrict 2
    ./stl10_single.sh --augment --S 1 --deltaorth --model diffopwrn16_8_stl --N 4 --restrict 2
    ./stl10_single.sh --augment --S 1 --deltaorth --model diffopwrn16_8_stl --restrict 3
    ./stl10_single.sh --augment --S 1 --deltaorth --model gausswrn16_8_stl --restrict 3
    ./stl10_single.sh --augment --S 1 --deltaorth --model gausswrn16_8_stl --N 4 --restrict 2
    ./stl10_single.sh --augment --S 1 --deltaorth --model e2wrn16_8_stl --N 8 --restrict 3
    ./stl10_single.sh --augment --S 1 --deltaorth --model e2wrn16_8_stl --N 4 --restrict 2
done