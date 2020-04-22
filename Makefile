BIN = out_delay_sparse_cocoa
CXX = mpiCC
CFLAGS = -O3 -g -Wall -std=c++11
LDFLAGS = 

all: $(BIN)

OBJS3 = delay_cocoa_sparse_main.o

out_delay_sparse_cocoa : $(OBJS3)
	$(CXX) $(LDFLAGS) -o out_delay_sparse_cocoa $(OBJS3) $(CFLAGS)

delay_cocoa_sparse_main.o : 
	$(CXX) $(CFLAGS) -c delay_cocoa_sparse_main.cpp 

clean :
	rm -f $(BIN)  $(OBJS3) 
