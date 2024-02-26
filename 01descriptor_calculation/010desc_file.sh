#!/bin/bash

#1st argument is the directory that contains all subdirectories with the volsite output
directory=$1

#2nd argument is the (name of) the csv file
csv_file=$2

#loop over the subdirectories to obtain 
for f in ${directory}/*
do
python3 main.py $f $2
done