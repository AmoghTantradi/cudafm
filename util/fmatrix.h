#ifndef FMATRIX_H_
#define FMATRIX_H_
#include <cstdint>

typedef uint64_t uint64;
typedef uint32_t uint;


const uint FMATRIX_EXPECTED_FILE_ID = 2;

template <typename T> struct sparse_entry {
    uint id;
    T value;
};

template <typename T> struct sparse_row {
	sparse_entry<T>* data;
	uint size;
};

template <typename T> struct sparse_row_v {
	uint size;
	sparse_entry<T> data[];
};



#endif