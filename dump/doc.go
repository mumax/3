/*
 Binary output format.
 Header: all 64-bit words:
	magic: "#dump10\n"
 	label for "time" coordinate (8 byte string like "t" or "f")
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

//NewWriter(io.Writer)
// 	Header
//	// set members
//	Write(data) error // repeat
//	writeHeader()
//	writeData32()
//	closeFrame()
//
//Reader
//	Read() frame, error: EOF=ok, shortread=KO
//
//Header
//	T, Size, Label, CRC bool ...
//
//Frame
//	Header
//	data
//
