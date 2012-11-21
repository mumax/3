/*
 PTX assembly of CUDA code.
*/
package ptx

// maps file names (equal to function names) on code
var ptxcode = make(map[string]string)
