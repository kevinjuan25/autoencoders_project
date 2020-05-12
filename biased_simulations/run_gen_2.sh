#!/bin/bash

epochs=500

python cv_boundaries.py iter_1_biased/mb_traj_combined.dat \
iter_1_biased/iter_1_convnet_${epochs}.pkl \
--imgprefix iter_1_biased/iter_1 --dataprefix iter_1_biased/iter_1

python generate_next_iter.py
