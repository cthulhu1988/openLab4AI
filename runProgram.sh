#!/bin/bash

for s in {1..5}
do
  for ((x=0;x<100;x++))
  do
    cat iris-data.txt | shuf > temp.txt
    head -n $s temp.txt > test.txt
    head -n -$s temp.txt > train.txt
    ./kmeans.py $s $s train.txt test.txt >> iris-results-$s.txt
    #cat iris-data.txt | ./split.bash $s ./id3decision.py >> results-$s.txt;
    #cat cancer-data.txt | ./split.bash $s ./id3.py >> cancer-results-$s.txt;
  done
done

# for s in {1,149}
# do
#   for ((x=0;x<10;x++))
#   do
#     cat iris-data.txt | shuf > temp.txt
#     head -n $s temp.txt > test.txt
#     head -n -$s temp.txt > train.txt
#
#     ./id3decision.py train.txt test.txt >> results-$s.txt
#     #cat iris-data.txt | ./split.bash $s ./id3decision.py >> results-$s.txt;
#   done
# done

#for s in {1,5,10,25,50,75,100,125,140,145,149}; do for ((x=0;x<10;x++)); do cat iris-data.txt| ./split.bash $s ./id3decision.py >> results-$s.txt; done; done
