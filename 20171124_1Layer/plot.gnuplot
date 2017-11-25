set datafile separator ","
set terminal png
set xlabel "steps"
set ylabel "accuracy"
set output 'output.png'
plot "output.csv" using 1:3 w l title "train dataset" ,"output.csv" using 1:4 w l title "valid dataset"