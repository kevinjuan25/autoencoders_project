#!/bin/bash

for i in {0..3}
do
    cp mb_biased.py iter_2_biased/${i}/
    cd iter_2_biased/${i}
    python mb_biased.py
    cd ../..
done
