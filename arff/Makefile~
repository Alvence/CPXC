#
# Main file responsible for compiling the source code and generating
# doxygen documentation
#

SRCDIR   := src
SRCFILES := $(shell find $(SRCDIR) -name "*.cpp")
OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))

CPP      := g++
CPPFLAGS := -g -Wall
INCLUDE  := -I$(SRCDIR)
LD       := g++
LDFLAGS  := -g -Wall -shared
AR       := ar
ARFLAGS  := rsv

LIBRARY  := arff
STATIC   := lib$(LIBRARY).a



all: static 

### static library ###
static: $(STATIC)

$(STATIC): $(OBJFILES)
	$(AR) $(ARFLAGS) $@ $^
###


### mostly generic ###
%.o: %.cpp
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cc
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c -o $@ $<

clean:
	rm -f $(STATIC) $(OBJFILES)
###	rm -f $(TEST) $(TESTOBJS) $(GTOBJS) $(GTLIB)
###
