# get name of this directory
DIR.NAME=$(shell basename $$PWD)
#
# fortran compiler
FC := gfortran
#
# general flags
compile = -c $<
output = -o $@
#
# options
options = -fimplicit-none
warnings = -W -Wall -Warray-temporaries -Wcharacter-truncation -Wextra				\
		-Wfatal-errors -Wfrontend-loop-interchange -Wimplicit-interface								\
		-Wintrinsics-std -Wsurprising -Wuninitialized -pedantic
debug = -g -fbacktrace -ffpe-trap=invalid,zero,overflow,underflow,denormal
#
# additional options for gfortran v4.5 and later
options_new = -std=f2018 -pedantic-errors
warnings_new = -Wimplicit-procedure -Winteger-division -Wreal-q-constant	\
-Wrealloc-lhs-all -Wuse-without-only
debug_new = -fcheck=all
# the following warnings may flood the screen and thereby obscure more important warnings
warnings_extra += -Wconversion-extra 
#
# concatenate options
options := $(options) $(options_new)
warnings := $(warnings) $(warnings_new)
debug := $(debug) $(debug_new)
#
# fortran compiler flags
FCFLAGS = $(includes) $(options) $(warnings) $(debug)
F77.FLAGS = -fd-lines-as-comments -fall-intrinsics
F90.FLAGS =
FC.COMPILE = $(FC) $(FCFLAGS) $(compile)
FC.COMPILE.o = $(FC.COMPILE) $(output) $(F77.FLAGS)
FC.COMPILE.o.f90 = $(FC.COMPILE) $(output) $(F90.FLAGS)
FC.COMPILE.mod = $(FC.COMPILE) -o $(OBJDIR)/$*.o $(F90.FLAGS)
#
# fortran linker flags
FLFLAGS = $(output) $^
FC.LINK = $(FC) $(FLFLAGS)
#
# define build directories
BINDIR := bin
MODDIR := mod
OBJDIR := obj
#
# source file lists
#
# program files (executable)
SRC.F77 = $(wildcard *.f)
SRC.F90 = $(wildcard *.f90)
# add SRCDIR if present
SRCDIR := src/fortran
ifneq ("$(strip $(wildcard $(SRCDIR)))","")
	VPATH += $(subst $(subst ,, ),:,$(strip $(SRCDIR)))
	SRC.F77 += $(wildcard $(SRCDIR)/*.f)
	SRC.F90 += $(wildcard $(SRCDIR)/*.f90)
endif
SRC = $(SRC.F77) $(SRC.F90)
#
# directory for "include" files (not executable, not compilable)
INCDIR := includes
# add INCDIR if present
ifneq ("$(strip $(wildcard $(INCDIR)))","")
	VPATH += $(subst $(subst ,, ),:,$(strip $(INCDIR)))
	includes = $(patsubst %,-I %,$(INCDIR))
	INCS.F77 = $(wildcard $(INCDIR)/*.f)
	INCS.F90 = $(wildcard $(INCDIR)/*.f90)
	INCS. +=  $(patsubst $(INCDIR)/%.f, %, $(INCS.F77)) \
	$(patsubst $(INCDIR)/%.f90, %, $(INCS.F90))
endif
#
# module files
# fortran module complier flags
FC.COMPILE.mod = $(FC.COMPILE) -o $(OBJDIR)/$*.o $(F90.FLAGS)
# source directory
MODDIR.in := modules
# add MODDIR.in if present
ifneq ("$(strip $(wildcard $(MODDIR.in)))","")
	VPATH += $(subst $(subst ,, ),:,$(strip $(MODDIR.in)))
	MODS.F77 = $(wildcard $(MODDIR.in)/*.f)
	MODS.F90 = $(wildcard $(MODDIR.in)/*.f90)
	MODS. +=  $(patsubst $(MODDIR.in)/%.f, %, $(MODS.F77)) \
	$(patsubst $(MODDIR.in)/%.f90, %, $(MODS.F90))
endif
# add additional modules
MODS. +=
# add MODDIR to includes if MODS. not empty
ifneq ("$(MODS.)","")
	includes:=$(includes) -J $(MODDIR)
endif
# build list of modules
MODS.mod = $(addsuffix .mod,$(MODS.))
MODS := $(addprefix $(MODDIR)/,$(MODS.mod))
#
# Add any external procudures below. Note: shared procedures should be included in a module
# unless written in a different language.
#
# function files
FUNDIR := functions
# add FUNDIR if present
ifneq ("$(strip $(wildcard $(FUNDIR)))","")
	VPATH += $(subst $(subst ,, ),:,$(strip $(FUNDIR)))
	FUNS.F77 = $(wildcard $(FUNDIR)/*.f)
	FUNS.F90 = $(wildcard $(FUNDIR)/*.f90)
	FUNS. +=  $(patsubst $(FUNDIR)/%.f, %, $(FUNS.F77)) \
	$(patsubst $(FUNDIR)/%.f90, %, $(FUNS.F90))
endif
# add additional fucntions
FUNS. +=
#
# subroutine files
SUBDIR := subroutines
# add SUBDIR if present
ifneq ("$(strip $(wildcard $(SUBDIR)))","")
	VPATH += $(subst $(subst ,, ),:,$(strip $(SUBDIR)))
	SUBS.F77 = $(wildcard $(SUBDIR)/*.f)
	SUBS.F90 = $(wildcard $(SUBDIR)/*.f90)
	SUBS. +=  $(patsubst $(SUBDIR)/%.f, %, $(SUBS.F77)) \
	$(patsubst $(SUBDIR)/%.f90, %, $(SUBS.F90))
endif
# add additional subroutines
SUBS. +=
#
# concatonate procedure lists (non-executables)
DEPS. = $(MODS.) $(SUBS.) $(FUNS.)
#
# build object lists
OBJS.F77 = $(SRC.F77:.f=.o)
OBJS.F90 = $(SRC.F90:.f90=.o)
OBJS.all = $(OBJS.F77) $(OBJS.F90)
OBJS.all := $(OBJS.all:$(SRCDIR)/%=%)
#
DEPS.o = $(addsuffix .o,$(DEPS.))
OBJS.o = $(filter-out $(DEPS.o),$(OBJS.all))
DEPS := $(addprefix $(OBJDIR)/,$(DEPS.o))
OBJS := $(addprefix $(OBJDIR)/,$(OBJS.o))
#
# executables
TARGET = 
EXES = $(addprefix $(BINDIR)/,$(OBJS.o:.o=))
#
# sub-programs
SUBDIRS := $(wildcard pi*)
#
# recipes
all: $(TARGET) $(EXES) $(SUBDIRS)
	@/bin/echo -e "\n$(DIR.NAME) $@ done"
$(SUBDIRS):
	@$(MAKE) --no-print-directory -C $@
printvars:
	@echo
	@echo "printing variables..."
	@echo "----------------------------------------------------"
	@echo
	@echo "DIR.NAME = '$(DIR.NAME)'"
	@echo "includes = '$(includes)'"
	@echo "VPATH = '$(VPATH)'"

	@echo
	@echo "----------------------------------------------------"
	@echo


	@echo "SUBDIRS = $(SUBDIRS)"

	@echo
	@echo "----------------------------------------------------"
	@echo

	@echo "SRC.F77 = $(SRC.F77)"
	@echo
	@echo "SRC.F90 = $(SRC.F90)"
	@echo
	@echo "SRC = $(SRC)"
	@echo
	@echo "OBJS.all = $(OBJS.all)"

	@echo
	@echo "----------------------------------------------------"
	@echo

	@echo "INCS. = $(INCS.)"
	@echo
	@echo "MODS. = $(MODS.)"
	@echo
	@echo "SUBS. = $(SUBS.)"
	@echo
	@echo "FUNS. = $(FUNS.)"
	@echo
	@echo "DEPS. = $(DEPS.)"

	@echo
	@echo "----------------------------------------------------"
	@echo

	@echo "OBJS.o = $(OBJS.o)"
	@echo
	@echo "DEPS.o = $(DEPS.o)"
	@echo
	@echo "MODS.mod = $(MODS.mod)"

	@echo
	@echo "----------------------------------------------------"
	@echo

	@echo "EXES = $(EXES)"
	@echo
	@echo "OBJS = $(OBJS)"
	@echo
	@echo "DEPS = $(DEPS)"
	@echo
	@echo "MODS = $(MODS)"
	@echo
	@echo "----------------------------------------------------"
	@echo "$@ done"
	@echo
#
# specific recipes
$(BINDIR)/ar: $(OBJDIR)/ar.o | $(BINDIR)
	@echo "compiling specific executable $@..."
	$(FC.LINK)
#
# generic recipes
$(BINDIR)/%: $(OBJDIR)/%.o $(DEPS) | $(BINDIR)
	@/bin/echo -e "\nlinking generic executable $@..."
	$(FC.LINK)
$(OBJDIR)/%.o: %.f $(MODS) | $(OBJDIR)
	@/bin/echo -e "\ncompiling generic object $@..."
	$(FC.COMPILE.o)
$(OBJDIR)/%.o: %.f90 $(MODS) | $(OBJDIR)
	@/bin/echo -e "\ncompiling generic f90 object $@..."
	$(FC.COMPILE.o.f90)
$(MODDIR)/%.mod: %.f | $(MODDIR)
	@/bin/echo -e "\ncompiling generic module $@..."
	$(FC.COMPILE.mod)
$(MODDIR)/%.mod: %.f90 | $(MODDIR)
	@/bin/echo -e "\ncompiling generic f90 module $@..."
	$(FC.COMPILE.mod)
#
# define directory creation
$(BINDIR):
	@echo "making $(BINDIR)..."
	@mkdir -pv $(BINDIR)
$(OBJDIR):
	@echo "making $(OBJDIR)..."
	@mkdir -pv $(OBJDIR)
$(MODDIR):
ifeq ("$(wildcard $(MODS))",)
	@echo "no modules specified"
else
	@echo "making $(MODDIR)..."
	@mkdir -pv $(MODDIR)
endif
#
# keep intermediate object files
.SECONDARY: $(DEPS) $(OBJS) $(MODS)
#
# recipes without outputs
.PHONY: all $(SUBDIRS) mostlyclean clean force out realclean distclean reset
#
# clean up
optSUBDIRS = $(addprefix $(MAKE) $@ --no-print-directory -C ,$(addsuffix ;,$(SUBDIRS)))
RM = @rm -vfrd
mostlyclean:
# remove compiled binaries
	@echo "removing compiled binary files..."
	$(RM) $(OBJDIR)/*.o
	$(RM) $(OBJDIR)
	$(RM) *.o *.obj
	$(RM) $(MODDIR)/*.mod
	$(RM) $(MODDIR)
	$(RM) *.mod
	$(RM) fort.*
	@$(optSUBDIRS)
	@echo "$(DIR.NAME) $@ done"
clean: mostlyclean
# remove binaries and executables
	@/bin/echo -e "\nremoving compiled executable files..."
	$(RM) $(BINDIR)/*
	$(RM) $(BINDIR)
	$(RM) *.exe
	$(RM) *.out
	@$(optSUBDIRS)
	@echo "$(DIR.NAME) $@ done"
force: clean
# force re-make
	@$(MAKE) --no-print-directory
out:
# remove outputs produced by executables
	@/bin/echo -e "\nremoving output files..."
	$(RM) fname*.in
	$(RM) svp.out
	$(RM) svp.in
	$(RM) state
	$(RM) test
	$(RM) test?
	@$(optSUBDIRS)
	@echo "$(DIR.NAME) $@ done"
realclean: clean out
# remove binaries and outputs
	@$(optSUBDIRS)
distclean: realclean
# remove binaries, outputs, and backups
	@/bin/echo -e "\nremoving backup files..."
# remove Git versions
	$(RM) *.~*~
# remove Emacs backup files
	$(RM) *~ \#*\#
# clean sub-programs
	@$(optSUBDIRS)
	@echo "$(DIR.NAME) $@ done"
reset: distclean
# remove untracked files
	@/bin/echo -e "\nresetting repository..."
	git reset HEAD
	git stash
	git clean -f
	@$(optSUBDIRS)
	@echo "$(DIR.NAME) $@ done"
#
# test the makefile
test: distclean printvars all
	@echo "$(DIR.NAME) $@ done"
#
# run executables
run: all
# run executables which do no require user input
	$(addprefix ./$(BINDIR)/,\
	ar \
	extrema \
	fmt \
	fun \
	global \
	globsubs \
	hello \
	huge \
	io \
	make_svp \
	test_getunit \
	sign \
	subs \
	sys \
	test_abs \
	test_system_clock \
	timedate \
	units \
	gethost )
	@$(optSUBDIRS)

run_man: all # test all functions that require manual input
	$(addprefix ./$(BINDIR)/,\
	ask \
	collatz \
	collatz_glide \
	fundem \
	pause )

run_int: all # test all functions that require user interrupt
	$(addprefix ./$(BINDIR)/,\
	collatz_loop \
	interrupt \
	timer )

run_fmt: all # test all functions that require set_fmt.f
	$(addprefix ./$(BINDIR)/,\
	collatz
	collatz_loop \
	fmt \
	huge \
	test_system_clock ))
