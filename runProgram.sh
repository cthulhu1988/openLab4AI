#!/bin/bash

for s in {1..2}
do
  for ((x=0;x<100;x++))
  do
    cat iris-data.txt | shuf > temp.txt
    head -n 10 temp.txt > test.txt
    head -n -10 temp.txt > train.txt
    ./kmeans.py $x $s train.txt test.txt >> iris-results-$s.txt
  done
done
