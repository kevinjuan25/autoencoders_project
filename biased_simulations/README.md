# Instructions to use code

For help with command line arguments, run the relevant .py file with Python
using the flag `--help`. For example
```
python train_autoencoder.py --help
python plots.py --help
```

## First round of CV discovery (unbiased)
- Copy mb_unbiased.py to relevant calculation folder.
```
cp mb_unbiased.py iter_0_unbiased
```

- Generate 1000-step unbiased MB trajectory by running
```
cd iter_0_unbiased
python mb_unbiased.py
```

- Train autoencoder on generated MB trajectory for N epochs with a batch size of M.
For example, train for 500 epochs and batch size 128 by running
```
python train_autoencoder.py iter_0_unbiased/mb_traj.dat 100 128 \
--outfile iter_0_unbiased/iter_0_convnet_100.pkl
```
This produces an AE object as `iter_0_convnet_100.pkl`, which can be loaded using pickle.

- Plot train and test losses for trained autoencoder
```
python plots.py --file iter_0_unbiased/iter_0_convnet_100.pkl \
--lossoutfile iter_0_unbiased/loss_history_100.png
```

- If more training is needed, restart from previous trained state. For example, to
train 100-epoch model for 400 more epochs to get a 500-epoch model, run
```
python train_autoencoder.py iter_0_unbiased/mb_traj.dat 500 128 \
--outfile iter_0_unbiased/iter_0_convnet_500.pkl \
--restart --restartfile iter_0_unbiased/iter_0_convnet_100.pkl
```

- Sometimes the optimizer can get stuck in a metastable state and not converge to the best
solution. Therefore, multiple rounds of training may be neccessary. Train until the best model fit is obtained.

- Using the trained autoencoder, generate
    - Binned CV histogram, p-values, and v-values (defined in Ferguson and Chen under boundary detection) for
    boundary detection
    - Trajectory, CV-reconstructed trajectory, and new bias point plot (bias points must be defined in code
    by user, based on histogram plots)
```
python cv_boundaries.py iter_0_unbiased/mb_traj.dat \
iter_0_unbiased/iter_0_convnet_500.pkl \
--imgprefix iter_0_unbiased/iter_0 --dataprefix iter_0_unbiased/iter_0
```

## Subsequent rounds of CV discovery

- Make relevant directory
- Modify `generate_next_iter.py` with relevant directory info and new bias points
- Run
```
python generate_next_iter.py
```
- Copy biased sampling code into folder, update location of umbrella.pt, perform biased sampling using similar steps to unbiased round
