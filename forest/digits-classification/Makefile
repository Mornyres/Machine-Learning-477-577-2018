# modified 1/31/16 by Chris Martin
# make flash now finds where STM board is mounted and
# directly copies the TARGET.bin without the need for stlink
TARGET = test

OUTPUTS = model.yml

OBJS   = train_digits.cpp 
CXX 	 = g++

LIBS=  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_ml

CXXFLAGS = -Wall -Wextra 

.PHONY : all

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX)  -o $(TARGET) $(CXXFLAGS) $(OBJS) $(LIBS)

clean:
	rm -f $(TARGET) $(OUTPUTS)
