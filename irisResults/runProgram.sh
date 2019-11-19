#!/bin/bash

for s in {35..140}
do
  for ((x=0;x<100;x++))
  do
    # cat iris-data.txt | shuf > temp.txt
    # head -n 10 temp.txt > test.txt
    # head -n -10 temp.txt > train.txt
    # ./kmeans.py $x $s train.txt test.txt >> iris-results-$s.txt
    cat iris-data.txt | ./split.bash 10 ./kmeans.py $x $s >> iris-results-$s.txt
  done
done
