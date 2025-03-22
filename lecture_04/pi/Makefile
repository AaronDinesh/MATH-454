# Compiler and Linker (defaults to g++, can be overridden)
CXX ?= g++
LD := $(CXX)

# The Target Binary Program (defaults to main)
MAIN ?= main
TARGET := build/$(MAIN)

# The Directories, Source, Includes, Objects, and Defines
BUILD_DIR := build
SRCEXT := cc
SOURCES := $(shell find . -type f -name '$(MAIN).$(SRCEXT)')
OBJECTS := $(patsubst ./%, $(BUILD_DIR)/%, $(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -Wall -O3 -std=c++17
INC := 
LIB := 
LDFLAGS := $(LIB)

# Default Make
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(LD) -o $@ $^ $(LDFLAGS)
	@rm -f $(OBJECTS)

# Compile
$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(INC) -c -o $@ $<
# Clean only objects and the binary
clean:
	rm -rf $(BUILD_DIR)

# Non-File Targets
.PHONY: all clean

