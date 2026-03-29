CXX = mpic++

TARGETS = brute bfs row_pipeline sqrt recursive

all: $(TARGETS)

%: %.cpp
	$(CXX) $< -o $@

clean:
	rm -f $(TARGETS)
