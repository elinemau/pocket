#!/bin/bash

#1st argument is the directory that contains all subdirectories with the volsite output
directory=$1

#2nd argument is the (name of) the csv file
csv_file=$2

#loop over the subdirectories to obtain 
for f in ${directory}/*
do
    pre_f=${f%.*}
    f_id=${pre_f##*/}
    if [[ $f = *novolsite* ]]
    then
        echo "The structure $f_id has no VolSite data to be run."
    else
        python3 code/pocket/main.py $f $2
        echo "The structure $f_id has been added to the descriptor csv file."
    fi
done