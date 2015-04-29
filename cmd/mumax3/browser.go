package main

import (
	"fmt"
	"os/exec"
)

// Try to open url in a browser. Instruct to do so if it fails.
func openbrowser(url string) {
	for _, cmd := range browsers {
		err := exec.Command(cmd, url).Start()
		if err == nil {
			fmt.Println("//openend web interface in", cmd)
			return
		}
	}
	fmt.Println("//please open ", url, " in a browser")
}

// list of browsers to try.
var browsers = []string{"x-www-browser", "google-chrome", "chromium-browser", "firefox", "explorer"}
