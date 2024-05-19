# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/antonio/Faculdade/SPD/Projeto1_SPD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/antonio/Faculdade/SPD/Projeto1_SPD

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/antonio/Faculdade/SPD/Projeto1_SPD/CMakeFiles /home/antonio/Faculdade/SPD/Projeto1_SPD//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/antonio/Faculdade/SPD/Projeto1_SPD/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named projeto1

# Build rule for target.
projeto1: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 projeto1
.PHONY : projeto1

# fast build rule for target.
projeto1/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/build
.PHONY : projeto1/fast

main.o: main.cpp.o
.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i
.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/main.cpp.s
.PHONY : main.cpp.s

src/parallel.o: src/parallel.cpp.o
.PHONY : src/parallel.o

# target to build an object file
src/parallel.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/parallel.cpp.o
.PHONY : src/parallel.cpp.o

src/parallel.i: src/parallel.cpp.i
.PHONY : src/parallel.i

# target to preprocess a source file
src/parallel.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/parallel.cpp.i
.PHONY : src/parallel.cpp.i

src/parallel.s: src/parallel.cpp.s
.PHONY : src/parallel.s

# target to generate assembly for a file
src/parallel.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/parallel.cpp.s
.PHONY : src/parallel.cpp.s

src/sequential.o: src/sequential.cpp.o
.PHONY : src/sequential.o

# target to build an object file
src/sequential.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/sequential.cpp.o
.PHONY : src/sequential.cpp.o

src/sequential.i: src/sequential.cpp.i
.PHONY : src/sequential.i

# target to preprocess a source file
src/sequential.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/sequential.cpp.i
.PHONY : src/sequential.cpp.i

src/sequential.s: src/sequential.cpp.s
.PHONY : src/sequential.s

# target to generate assembly for a file
src/sequential.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/sequential.cpp.s
.PHONY : src/sequential.cpp.s

src/util.o: src/util.cpp.o
.PHONY : src/util.o

# target to build an object file
src/util.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/util.cpp.o
.PHONY : src/util.cpp.o

src/util.i: src/util.cpp.i
.PHONY : src/util.i

# target to preprocess a source file
src/util.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/util.cpp.i
.PHONY : src/util.cpp.i

src/util.s: src/util.cpp.s
.PHONY : src/util.s

# target to generate assembly for a file
src/util.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/projeto1.dir/build.make CMakeFiles/projeto1.dir/src/util.cpp.s
.PHONY : src/util.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... projeto1"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... src/parallel.o"
	@echo "... src/parallel.i"
	@echo "... src/parallel.s"
	@echo "... src/sequential.o"
	@echo "... src/sequential.i"
	@echo "... src/sequential.s"
	@echo "... src/util.o"
	@echo "... src/util.i"
	@echo "... src/util.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

