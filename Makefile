CXX = emcc
CXXFLAGS = -O3 -msimd128 -flto -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS="['_init_model','_forward_step','_free_model']"

all: dist/narrative_core.js

dist/narrative_core.js: cpp/narrative_core.cpp
	mkdir -p dist
	$(CXX) $(CXXFLAGS) -o dist/narrative_core.js cpp/narrative_core.cpp

clean:
	rm -rf dist
