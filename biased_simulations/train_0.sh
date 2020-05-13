#!/bin/bash

epochs=3000

python train_autoencoder.py iter_0_unbiased/mb_traj.dat ${epochs} 128 \
--outfile iter_0_unbiased/iter_0_convnet_${epochs}.pkl \
#--restart --restartfile iter_0_unbiased/iter_0_convnet_${prevepochs}.pkl

python plots.py --file iter_0_unbiased/iter_0_convnet_${epochs}.pkl \
--lossoutfile iter_0_unbiased/loss_history_${epochs}.png

python cv_boundaries.py iter_0_unbiased/mb_traj.dat \
iter_0_unbiased/iter_0_convnet_${epochs}.pkl \
--imgprefix iter_0_unbiased/iter_0_${epochs} --dataprefix iter_0_unbiased/iter_0_${epochs} \
--skipanimate
