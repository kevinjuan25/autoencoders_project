#!/bin/bash

epochs=2000

python cv_boundaries.py iter_1_biased/mb_traj_combined.dat \
iter_1_biased/iter_1_convnet_${epochs}.pkl \
--imgprefix iter_1_biased/iter_1_${epochs} --dataprefix iter_1_biased/iter_1_${epochs}

python generate_next_iter.py
