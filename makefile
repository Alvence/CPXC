#
# Main file responsible for compiling the source code
# By Yunzhe(Alvin) Jia
# 2016-01-18, The University of Melbourne
#

SRCDIR   := .
ARFFSRCDIR := arff/src

ARFFLIBDIR := arff
ARFFLIB	 := arff

SRCFILES := $(shell find $(SRCDIR) -name "*.cc")
#OBJFILES := $(patsubst %.cc,%.o,$(SRCFILES))
OBJFILES := BinDivider.o CP.o

CC      := g++
CFLAGS := -g -Wall
LFLAGS := -L$(ARFFLIBDIR) -l$(ARFFLIB) -LDPM -lDPM `pkg-config --cflags --libs opencv`
INCLUDE  := -I$(SRCDIR) -I$(ARFFSRCDIR) -IDPM
LD       := g++
LDFLAGS  := -g -Wall -shared

TARGET   := main

all: $(OBJFILES) $(TARGET) 

### target ###
$(TARGET): $(OBJECTS) main.cc
	$(CC) $(CFLAGS)  $(INCLUDE) $(OBJFILES)  -o $@ $<  $(LFLAGS)

test: arff_test.cc 
	$(CC) $(CFLAGS)  $(INCLUDE) -o $@ $<  $(LFLAGS)

### mostly generic ###
%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $< $(LFLAGS)

#clean:
#	rm -f $(STATIC) $(OBJFILES)
###	rm -f $(TEST) $(TESTOBJS) $(GTOBJS) $(GTLIB)
###
clean:
	rm -R temp/ *.o *~ main test *.dSYM	
