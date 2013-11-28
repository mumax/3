package mainpkg

import "github.com/mumax/3/engine"

func Main() {
	Init()
	defer engine.Close()
	RunFiles()
}
