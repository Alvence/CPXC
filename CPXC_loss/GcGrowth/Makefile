#make file by Yunzhe(Alvin) Jia, 
#2016-04-13, The University of Melbourne

OBJS = gcgrowth.o ScanDBMine.o parameters.o PatternSet.o FPtree.o Global.o data.o fsout.o
CC = g++
AR = ar
ARFLAG = -rsv
LIB = libGcGrowth.a
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

LIB_PATH = /usr/lib
INCLUDE_PATH = /usr/include

all: $(LIB)

$(LIB): $(OBJS)
	@rm -f $(LIB)
	$(AR) $(ARFLAG) $(LIB) $(OBJS)
	@rm *.o

Global.o: Global.cpp Global.h
	$(CC) -c $(CFLAGS) Global.cpp
	
ScanDBMine.o : ScanDBMine.cpp ScanDBMine.h Global.h FPtree.h data.h inline_routine.h
	$(CC) -c $(CFLAGS) ScanDBMine.cpp

FPtree.o : FPtree.cpp FPtree.h inline_routine.h
	$(CC) -c $(CFLAGS) FPtree.cpp
	
data.o : data.cpp data.h
	$(CC) -c $(CFLAGS) data.cpp

fsout.o: fsout.cpp fsout.h Global.h parameters.h 
	$(CC) -c $(CFLAGS) fsout.cpp

parameters.o: parameters.cpp parameters.h Global.h
	$(CC) -c $(CFLAGS) parameters.cpp

PatternSet.o: PatternSet.cpp PatternSet.h FPtree.h Global.h inline_routine.h fsout.h
	$(CC) -c $(CFLAGS) PatternSet.cpp

gcgrowth.o : gcgrowth.cpp ScanDBMine.h FPtree.h data.h fsout.h Global.h parameters.h inline_routine.h
	$(CC) -c $(CFLAGS) gcgrowth.cpp

install:
	@cp $(LIB) $(LIB_PATH)/
	@cp *.h $(INCLUDE_PATH)/gcgrowth/

clean:
	rm *.o *.a

