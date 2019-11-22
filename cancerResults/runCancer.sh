#!/bin/bash

for s in {96..110}
do
  cat cancer-data.txt | shuf > temp-$s.txt
  head -n 10 temp-$s.txt > test-$s.txt
  head -n -10 temp-$s.txt > train-$s.txt
  for ((x=0;x<100;x++))
  do
     ./kmeans.py $x $s train-$s.txt test-$s.txt >> cancer-results-$s.txt &
  done
done
