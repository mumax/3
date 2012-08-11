/*
 Binary output format.
 Header contains all 64-bit words. Strings are 8 bytes long.
	Magic number "#dump002"
	Components int
	MeshSize   [3]int
	MeshStep   [3]float64
	MeshUnit   string
	Time       float64
	TimeUnit   string
	DataLabel  string
	DataUnit   string
	Precission uint64
 	DATA
 	crc64 of DATA and header using ISO polynomial 0xD800000000000000.
*/
package dump
