/*
 Package nimble provides the basic user API for building simulations.

 NOTICE:
 	pre-alpha stage, API may change regularly.

 TODO:  Chan.Buffer() buffers one frame, if not yet so.
 TODO:  Consistent API:
	Only Chan interfaces should be used
 	NewXXX() does not Stack()
 	RunXXX() Stacks, returns output(s) ? (to avoid confusion with New)
 	ExecUnsafe()
	Run()
 TODO: Automatically report run time.
*/
package nimble
