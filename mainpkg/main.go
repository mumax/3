package mainpkg

import "github.com/mumax/3/engine"

func Main() {
	Init()
	defer prof.Cleanup()
	defer engine.Close()
	RunFiles()
}
