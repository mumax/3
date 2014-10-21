package engine

import "github.com/mumax/3/util"

var allowUnsafe = false // allow unsafe features?

func init() {
	DeclFunc("ext_EnableUnsafe", EnableUnsafe, "Allow potentially unsafe features, at your own risk.")
}

func EnableUnsafe() {
	util.Log("Allowing unsafe features")
	allowUnsafe = true
}
