do:
	python3 normal.py
output:
	python3 -u normal.py | tee output.csv
output.png:output.csv plot.gnuplot
	gnuplot -c plot.gnuplot