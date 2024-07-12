CC=g++

linear: main.cpp
	$(CC) main.cpp linear_sol.cpp ../matrix.cpp -o main
	./main < linear_dataset.csv > out.txt
clean:
	rm main