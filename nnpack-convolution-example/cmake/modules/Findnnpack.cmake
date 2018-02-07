
MESSAGE(STATUS "USING BUNDLED Findnnpack.cmake ...")
FIND_PATH(
	NNPACK_INCLUDE_DIRS
	nnpack.h
	/usr/local/include/nnpack
	/usr/local/include/)

FIND_LIBRARY(
	NNPACK_LIBRARAIES NAMES nnpack
	PATHS /usr/local/lib/)
