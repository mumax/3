#ifndef _MASK_H_
#define _MASK_H_

// a mask is interpreted as array[i], or 1 if array is NULL
#define loadmask(array, index) (array == NULL? 1.0f: array[(index)])

#endif
