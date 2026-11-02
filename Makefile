CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra
PKGS    := wayland-client cairo
CFLAGS  += $(shell pkg-config --cflags $(PKGS)) -Igen
LDLIBS  += $(shell pkg-config --libs $(PKGS)) -lm

PROTOS  := xdg-shell color-management-v1 xdg-decoration-unstable-v1
GEN_H   := $(PROTOS:%=gen/%-client-protocol.h)
GEN_C   := $(PROTOS:%=gen/%-protocol.c)

scrgb-test: main.c $(GEN_C) $(GEN_H)
	$(CC) $(CFLAGS) -o $@ main.c $(GEN_C) $(LDLIBS)

gen/%-client-protocol.h: protocols/%.xml
	@mkdir -p gen
	wayland-scanner client-header $< $@

gen/%-protocol.c: protocols/%.xml
	@mkdir -p gen
	wayland-scanner private-code $< $@

clean:
	rm -rf gen scrgb-test

.PHONY: clean
