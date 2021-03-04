CC=g++
OPENCV_INSTALL_PATH=/home/peeth/Todor/opencv/install
LIBS_PATHS=-I my_headers/ -I $(OPENCV_INSTALL_PATH)/include
OPENCV_LIBS=`pkg-config opencv --libs`
CFLAGS=$(LIBS_PATHS) -c
LDFLAGS=-std=c++11 -O3 -g
SOURCES=Source_code/my_settings.cpp Source_code/glam3D.cpp Source_code/rlm3D.cpp Source_code/my_kmeans.cpp \
		Source_code/volume_metrics.cpp Source_code/fos3D.cpp Source_code/mitra.cpp Source_code/mitra_var_1.cpp \
		Source_code/mitra_var_2.cpp Source_code/Irrelevancy_Filter.cpp Source_code/clusterScatteringCriteria.cpp \
		Source_code/rank.cpp Source_code/sRank.cpp Source_code/FuzzyCMeans.cpp Source_code/glcm3D.cpp Source_code/dcSettings.cpp
OBJECTS=$(SOURCES:.cpp=.o)
SRC_PATH=Source_code/
SOURCES2=Source_code/clusterTextures.cpp Source_code/datasetCreator.cpp
OBJECTS2=$(SOURCES2:.cpp=.o)
EXECUTABLEST=$(SOURCES2:.cpp=)
EXECUTABLES=$(subst Source_code/,,$(EXECUTABLEST))

all: $(SOURCES) $(SOURCES2) $(EXECUTABLES)

print-%: ; @echo $* = $($*)

$(EXECUTABLES): $(OBJECTS2) $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(SRC_PATH)$@.o -o $@ -lpthread $(OPENCV_LIBS)
.cpp.o:
	$(CC) $(LDFLAGS) $(CFLAGS) $< -o $@

rm:
	rm $(EXECUTABLES) $(OBJECTS) $(OBJECTS2)
