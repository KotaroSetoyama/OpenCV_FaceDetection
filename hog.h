#ifndef _HOG_H_
#define _HOG_H_

#include <vector>

std::vector<float> GetHoG(unsigned char* img, int _SIZE_X, int _SIZE_Y, int _CELL_BIN = 9, int _CELL_X = 5, int _CELL_Y = 5, int _BLOCK_X = 3, int _BLOCK_Y = 3);

#endif