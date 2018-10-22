#!/bin/bash

# Script to apply algorithm to sequences
counter=3
while [ $counter -le 42 ]
do
   ./out.exe $counter 10000
   ((counter++))
done
echo All done

