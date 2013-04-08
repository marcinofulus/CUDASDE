all: prog1 prog2 prog3

clean:
	rm -f prog1 prog2 prog3 *.linkinfo

GSL_LIBS = $(shell gsl-config --libs)

prog1: prog1.cu 
	nvcc --use_fast_math $+ -o $@

prog2: prog2.cu 
	nvcc --use_fast_math $+ -o $@ $(GSL_LIBS)

prog3: prog3.cu 
	nvcc --use_fast_math $+ -o $@ $(GSL_LIBS)



