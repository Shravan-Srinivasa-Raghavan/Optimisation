CC=g++
CFLAGS=-mavx

all: linear logistic regression test

linear: ./linear/main.cpp ./linear/linear.cpp matrix.cpp threads.cpp ./linear/plotter.py
	$(CC) $(CFLAGS) ./linear/main.cpp ./linear/linear.cpp kernel.cpp matrix.cpp threads.cpp -o ./linear/main -std=c++20 -pthread -I./OpenCL/include -L./OpenCL/lib -lOpenCL
	./linear/main < ./linear/linear_dataset.csv > ./linear/out.txt
	python3 ./linear/plotter.py

logistic: ./logistic/main.cpp ./logistic/logistic.cpp matrix.cpp threads.cpp ./logistic/plotter.py
	$(CC) $(CFLAGS) ./logistic/main.cpp ./logistic/logistic.cpp kernel.cpp matrix.cpp threads.cpp -o ./logistic/main -std=c++20 -pthread -I./OpenCL/include -L./OpenCL/lib -lOpenCL
	./logistic/main < ./logistic/logistic_dataset.csv > ./logistic/out.txt
	python3 ./logistic/plotter.py

regression: ./regression/main.cpp ./regression/regression.cpp matrix.cpp threads.cpp ./regression/plotter.py
	$(CC) $(CFLAGS) ./regression/main.cpp ./regression/regression.cpp kernel.cpp matrix.cpp threads.cpp -o ./regression/main -std=c++20 -pthread -I./OpenCL/include -L./OpenCL/lib -lOpenCL
	./regression/main < ./regression/regression_dataset.csv > ./regression/out.txt
	python3 ./regression/plotter.py

test: matrix.cpp tester.cpp threads.cpp
	$(CC) $(CFLAGS) kernel.cpp tester.cpp matrix.cpp threads.cpp -o ./bin/test -std=c++20 -pthread -I./OpenCL/include -L./OpenCL/lib -lOpenCL
	./bin/test 10


make clean:
	rm ./bin/* ./linear/main ./logistic/main ./regression/main ./linear/*.png ./logistic/*.png ./regression/*.png