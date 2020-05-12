#!/bin/bash

epochs=500

python train_autoencoder.py iter_1_biased/mb_traj_combined.dat ${epochs} 128 \
--outfile iter_1_biased/iter_1_convnet_${epochs}.pkl \
#--restart --restartfile iter_0_unbiased/iter_0_convnet_${prevepochs}.pkl

python plots.py --file iter_1_biased/iter_1_convnet_${epochs}.pkl \
--lossoutfile iter_1_biased/loss_history_${epochs}.png

python cv_boundaries.py iter_1_biased/mb_traj_combined.dat \
iter_1_biased/iter_1_convnet_${epochs}.pkl \
--imgprefix iter_1_biased/iter_1 --dataprefix iter_1_biased/iter_1 \
--skipanimate
