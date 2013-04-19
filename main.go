package main

// main program starts a web gui.

import (
	. "code.google.com/p/mx3/engine"
	"log"
	"os/exec"
)

func main() {
	Init()
	defer Close()

	SetMesh(32, 32, 1, 5e-9, 5e-9, 5e-9)

	Msat = Const(1000e3)
	Aex = Const(10e-12)
	Alpha = Const(1)
	SetMUniform(1, 1, 0)

	go openbrowser("http://localhost" + *Flag_port)
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
