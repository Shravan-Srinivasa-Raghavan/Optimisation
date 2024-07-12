CC=g++

regression: main.cpp
	$(CC) main.cpp regression.cpp ../matrix/matrix.cpp -o main
	./main > out.txt
clean:
	rm main