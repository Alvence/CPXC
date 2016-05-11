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
OBJFILES := MLAlg.o BinDivider.o CP.o

CC      := g++
MAKE    := make
RM 			:= rm
CFLAGS := -g -w 
#LFLAGS := -L$(ARFFLIBDIR) -l$(ARFFLIB)  -LGcGrowth -lGcGrowth -lDPM -LDPM `pkg-config --cflags --libs opencv` -lm -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc 
#INCLUDE  := -I$(SRCDIR) -I$(ARFFSRCDIR) -IGcGrowh -IDPM
LFLAGS := -L$(ARFFLIBDIR) -l$(ARFFLIB)  -lDPM -LDPM `pkg-config --cflags --libs opencv` -lm -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc 
INCLUDE  := -I$(SRCDIR) -I$(ARFFSRCDIR) -IDPM
LD       := g++
LDFLAGS  := -g -Wall -shared
SUBDIRS = arff DPM GcGrowth
TARGET   := main


all: LIBS $(OBJFILES) $(TARGET) 

LIBS:
	$(MAKE) -C arff
	$(MAKE) -C DPM
	$(MAKE) -C GcGrowth

### target ###
$(TARGET): $(OBJECTS) main.cc
	$(CC) $(CFLAGS)  $(INCLUDE) $(OBJFILES)  -o $@ $<  $(LFLAGS)

test: $(OBJECTS) test.cc 
	$(CC) $(CFLAGS)  $(INCLUDE) $(OBJECTS)   -o $@ $<  $(LFLAGS)

### mostly generic ###
%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDE) -c -o $@ $< $(LFLAGS)

clean_sub: 
	for dir in $(SUBDIRS); do \
          $(MAKE) -C $$dir clean;\
	done
clean:
	rm -R  *.o *~ main test *.dSYM temp/*
