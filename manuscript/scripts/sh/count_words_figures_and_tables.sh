#!/bin/bash

# Table count
table_count=$(ls ./src/tables/2_legend*.tex 2>/dev/null | wc -l)
echo $table_count > ./src/counts/table_count.txt

# Figure count
figure_count=$(ls ./src/figures/2_legend/*.tex 2>/dev/null | wc -l)
echo $figure_count > ./src/counts/figure_count.txt

# IMRaD count
texcount ./src/abstract.tex -inc -1 -sum > ./src/counts/abstract_count.txt
texcount ./src/introduction.tex -inc -1 -sum > ./src/counts/introduction_count.txt
texcount ./src/methods.tex -inc -1 -sum > ./src/counts/methods_count.txt
texcount ./src/results.tex -inc -1 -sum > ./src/counts/results_count.txt
texcount ./src/discussion.tex -inc -1 -sum > ./src/counts/discussion_count.txt

# The sum of IMRaD count
(texcount ./src/introduction.tex -inc -1 -sum;
 texcount ./src/methods.tex -inc -1 -sum;
 texcount ./src/results.tex -inc -1 -sum;
 texcount ./src/discussion.tex -inc -1 -sum) | awk '{s+=$1} END {print s}' > ./src/counts/imrd_count.txt

## EOF
