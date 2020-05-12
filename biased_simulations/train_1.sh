#!/bin/bash

prevepochs=1000
epochs=1000
newepochs=2000

python train_autoencoder.py iter_1_biased/mb_traj_combined.dat ${epochs} 128 \
--outfile iter_1_biased/iter_1_convnet_${newepochs}.pkl \
--restart --restartfile iter_1_biased/iter_1_convnet_${prevepochs}.pkl

python plots.py --file iter_1_biased/iter_1_convnet_${newepochs}.pkl \
--lossoutfile iter_1_biased/loss_history_${newepochs}.png

python cv_boundaries.py iter_1_biased/mb_traj_combined.dat \
iter_1_biased/iter_1_convnet_${newepochs}.pkl \
--imgprefix iter_1_biased/iter_1_${newepochs} --dataprefix iter_1_biased/iter_1_${newepochs} \
--skipanimate
