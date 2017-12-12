do:

output1.csv:nlayer.py
	python3 -u nlayer.py 1 | tee output1.csv
output5.csv:nlayer.py
	python3 -u nlayer.py 5 | tee output5.csv
output10.csv:nlayer.py
	python3 -u nlayer.py 10 | tee output10.csv
output20.csv:nlayer.py
	python3 -u nlayer.py 20 | tee output20.csv
output.png:output1.csv output5.csv output10.csv output20.csv plot.gnuplot
	gnuplot -c plot.gnuplot
clean:
	rm -rf output*.csv output.png
