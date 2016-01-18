#make file by Yunzhe(Alvin) Jia, 
#2015-12-11, The University of Melbourne

OBJS = data.o Global.o parameters.o FPtree.o fsout.o PatternSet.o ScanDBMine.o DPM.o
CC = g++
AR = ar
ARFLAG = -rsv
LIB = libDPM.a
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

DPM.o : DPM.h DPM.cpp
	$(CC) $(LFLAGS) $(CFLAGS) DPM.cpp

data.o:data.h data.cpp
	$(CC) $(LFLAGS) $(CFLAGS) data.cpp

Global.o:Global.h Global.cpp
	$(CC) $(LFLAGS) $(CFLAGS) Global.cpp

parameters.o:Global.o parameters.h parameters.cpp
	$(CC) $(LFLAGS) $(CFLAGS) parameters.cpp

FPtree.o:inline_routine.h FPtree.h FPtree.cpp
	$(CC) $(LFLAGS) $(CFLAGS) FPtree.cpp

fsout.o:Global.o parameters.o fsout.h fsout.cpp
	$(CC) $(LFLAGS) $(CFLAGS) fsout.cpp

PatternSet.o:Global.o fsout.o inline_routine.h PatternSet.h PatternSet.cpp
	$(CC) $(LFLAGS) $(CFLAGS) PatternSet.cpp

ScanDBMine.o:Global.o FPtree.o data.o inline_routine.h ScanDBMine.h ScanDBMine.cpp
	$(CC) $(LFLAGS) $(CFLAGS) ScanDBMine.cpp

install:
	@cp $(LIB) $(LIB_PATH)/
	@cp *.h $(INCLUDE_PATH)/dpm/

clean:
	rm *.o *.a

