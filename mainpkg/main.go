package mainpkg

import (
	"github.com/mumax/3/engine"
	"github.com/mumax/3/prof"
)

func Main() {
	Init()
	defer prof.Cleanup()
	defer engine.Close()
	RunFiles()
}
