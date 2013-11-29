package engine

// declare functionality for interpreted input scripts

import "github.com/mumax/3/script"

// holds the script state (variables etc)
var World = script.NewWorld()

// Add a function to the script world
func DeclFunc(name string, f interface{}, doc string) {
	World.Func(name, f, doc)
}

// Add a constant to the script world
func DeclConst(name string, value float64, doc string) {
	World.Const(name, value, doc)
}

// Add a read-only variable to the script world.
// It can be changed, but not by the user.
func DeclROnly(name string, value interface{}, doc string) {
	World.ROnly(name, value, doc)
	GUI.Add(name, value)
}

// Add a (pointer to) variable to the script world
func DeclVar(name string, value interface{}, doc string) {
	World.Var(name, value, doc)
	GUI.Add(name, value)
}

// Add an LValue to the script world.
// Assign to LValue invokes SetValue()
func DeclLValue(name string, value script.LValue, doc string) {
	World.LValue(name, value, doc)
	GUI.Add(name, value)
}

// evaluate code, exit on error (behavior for input files)
func EvalFile(code *script.BlockStmt) {
	for i := range code.Children {
		LogInput(script.Format(code.Node[i]))
		code.Children[i].Eval()
	}
}

// evaluate code, report error (behaviour for GUI)
//func EvalGUI(code*script.BlockStmt){
//
//}
