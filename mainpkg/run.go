package mainpkg

import (
	"flag"
	"fmt"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"io/ioutil"
	"os/exec"
	"time"
)

func RunFiles() {
	if !*flag_vet {
		runFileAndServe(flag.Arg(0))
		keepBrowserAlive() // if open, that is
	}
}

// Runs a script file.
func runFileAndServe(fname string) {

	var code *script.BlockStmt
	var err2 error
	if fname != "" {
		// first we compile the entire file into an executable tree
		bytes, err := ioutil.ReadFile(fname)
		util.FatalErr(err)
		code, err2 = engine.World.Compile(string(bytes))
		util.FatalErr(err2)
	}

	// now the parser is not used anymore so it can handle web requests
	fmt.Print("starting GUI at http://localhost", *flag_port, "\n")
	go engine.Serve(*flag_port)

	if fname != "" {
		// start executing the tree, possibly injecting commands from web gui
		engine.EvalFile(code)
	} else {
		fmt.Println("no input files: starting interactive session")
		openbrowser("http://localhost" + *flag_port)
		keepBrowserAlive()
	}
}

func keepBrowserAlive() {
	if time.Since(engine.KeepAlive()) < engine.Timeout {
		fmt.Println("keeping session open to browser")
		go func() {
			for {
				engine.Inject <- nop // wake up RunInteractive so it may exit
				time.Sleep(1 * time.Second)
			}
		}()
		engine.RunInteractive()
	}
}

func nop() {}

// Try to open url in a browser. Instruct to do so if it fails.
func openbrowser(url string) {
	for _, cmd := range browsers {
		err := exec.Command(cmd, url).Start()
		if err == nil {
			fmt.Println("openend web interface in", cmd)
			return
		}
	}
	fmt.Println("Please open ", url, " in a browser")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "ie", "iexplore"}
