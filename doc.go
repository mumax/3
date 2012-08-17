/*
 Experimental finite difference time domain simulation package.
*/
package main

import (
	_ "nimble-cube/cli"
	_ "nimble-cube/core"
	_ "nimble-cube/dump"
	_ "nimble-cube/gpu"
	_ "nimble-cube/gpu/ptx"
	_ "nimble-cube/gpu/conv"
	_ "nimble-cube/mag"
	_ "nimble-cube/unit"
)

func main() {
	// just a dummy. this file exists only for documentation
}
