set datafile separator ","
set terminal png
set xlabel "steps"
set ylabel "accuracy"
set output 'output.png'
plot "output1.csv" using 2:5 w l title "1" ,"output5.csv" using 2:5 w l title "5","output10.csv" using 2:5 w l title "10","output20.csv" using 2:5 w l title "20"