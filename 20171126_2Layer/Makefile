do:
	python3 2layer.py
output.csv:2layer.py
	python3 2layer.py | tee output.csv
output.png:output.csv plot.gnuplot
	gnuplot -c plot.gnuplot