/*
 This is mx3's public API used to construct input scripts. Typical script:

 	package main

 	import . "code.google.com/p/mx3/engine"

 	func main() {
 		Init()
 		defer Close()

 		SetMesh(128, 32, 1, 3e-9, 3e-9, 20e-9)

 		// Material parameters
 		Msat = Const(600e3)
 		Aex = Const(10e-12)
 		Alpha = Const(0.02)

 		// Initial magnetization
 		M.Set(Uniform(1, 0, 0))

 		// Schedule output
 		M.Autosave(100e-12)
 		Table.Autosave(10e-12)

 		Run(1e-9)
 	}

 The examples/ directory contains more examples.

 To run the input script:
 	go run myinput.go -o outputdir
 The optional -o flag sets the output directory. For a list of all flags, see:
 	go run myinput.go -help
*/
package engine

// TODO: check example
