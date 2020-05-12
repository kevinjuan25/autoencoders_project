#!/bin/bash

epochs=5000

python cv_boundaries.py iter_0_unbiased/mb_traj.dat \
iter_0_unbiased/iter_0_convnet_${epochs}.pkl \
--imgprefix iter_0_unbiased/iter_0 --dataprefix iter_0_unbiased/iter_0

python generate_next_iter.py
