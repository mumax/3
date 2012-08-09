/*
 Binary output format.
 Header: all 64-bit words:
	magic number: "#dump10\n"
 	label for "time" coordinate (max 8 byte string like "t(s)" or "f(Hz)")
 	time of the snapshot (double)
 	label for "space" coordinate (like "r" or "k")
 	cellsize
	data rank: always 4
	4 sizes for each direction, like: 3  128 256 1024
 	Precission of data: 4 for float32, 8 for float64
 	DATA
 	crc64 of DATA and header using ISO polynomial 0xD800000000000000.
*/
package dump
