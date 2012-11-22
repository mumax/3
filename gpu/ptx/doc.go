/*
 PTX assembly of CUDA code.
*/
package ptx

// maps file names (equal to function names) on code
// TODO: do not use map but generate functions so that
// the linker can elide unused ptx code.
var Code = make(map[string]string)
