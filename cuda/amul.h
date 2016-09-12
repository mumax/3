#ifndef _AMUL_H_
#define _AMUL_H_

#define amul(arr, mul, i) \
		( (arr == NULL)? (mul): (mul * arr[i]) )

#define div(a, b) \
		( (b == 0.0f)? (0.0f): (a/b) )

#endif
