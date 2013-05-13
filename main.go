package main

import (
	. "code.google.com/p/mx3/engine"
	"code.google.com/p/mx3/util"
	"flag"
	"log"
	"os/exec"
)

// dummy imports to fetch those files
import (
	_ "code.google.com/p/mx3/examples"
	_ "code.google.com/p/mx3/test"
)

func main() {
	Init()
	defer Close()

	// flags parsed by engine.Init()
	if flag.NArg() > 0 {
		if *Flag_od == "" { // -o not set
			SetOD(util.NoExt(flag.Arg(0))+".out", *Flag_force)
		}
		for _, f := range flag.Args() {
			log.Println("run file", f)
			RunFile(f)
		}
	} else {
		log.Println("no input files: starting interactive session")
		interactive()
	}

}

//
func interactive() {
	SetMesh(32, 32, 1, 5e-9, 5e-9, 5e-9)
	Msat = Const(1000e3)
	Aex = Const(10e-12)
	Alpha = Const(1)
	M.Set(Uniform(1, 1, 0))
	RunInteractive()
}

// Try to open url in a browser. Instruct to do so if it fails.
func openbrowser(url string) {
	for _, cmd := range browsers {
		err := exec.Command(cmd, url).Start()
		if err == nil {
			log.Println("\n ====\n openend web interface in", cmd, "\n ====\n")
			return
		}
	}
	log.Println("\n ===== \n Please open ", url, " in a browser \n ==== \n")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}
