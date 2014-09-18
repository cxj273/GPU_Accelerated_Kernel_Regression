##################################################
##    Linear Regression                         ##
##################################################

CC=g++
NVCC=/usr/local/cuda-5.0/bin/nvcc
FC=gfortran
ifdef debug
CFLAGS=-g
else
CFLAGS=-DEBUG -O3 -arch=sm_30
endif
INCLUDES=-I/home/uqxchan1/.local/cula/include
LIBPATH32=-L/home/uqxchan1/.local/cula/lib
LIBPATH64=-L/home/uqxchan1/.local/cula/lib64
LIBS=-lcula_lapack -lcublas -liomp5

all: CrossValidation

CrossValidation	: main.o KR.o crossValidationKR.o
	$(NVCC) -o CrossValidation main.o KR.o crossValidationKR.o $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)

main.o	: main.cu KR.h crossValidationKR.h
	$(NVCC) -c main.cu $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)

KR.o	: KR.cu KR.h
	$(NVCC) -c KR.cu $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)

crossValidationKR.o		: crossValidationKR.cu crossValidationKR.h KR.h
	$(NVCC) -c crossValidationKR.cu $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)

clean:
	rm -f CrossValidation
	rm -rf *.o
