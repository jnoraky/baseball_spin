SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)

GCC = g++ -std=c++11 -O3
CXXFLAGS = `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`

out.exe: $(OBJ)
	$(GCC) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) tum.exe
