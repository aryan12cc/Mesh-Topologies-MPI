CXX = mpic++

TARGETS = naive_unserialized chunked_serialized

all: $(TARGETS)

%: %.cpp
	$(CXX) $< -o $@

clean:
	rm -f $(TARGETS) naive_serialized
