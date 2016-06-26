CC=gcc
CFLAGS=-O3 -c -Wall
SOURCES=naive_bayes.c main.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ -lm -lgsl -lgslcblas
	rm  $(OBJECTS)
.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm  $(EXECUTABLE)
