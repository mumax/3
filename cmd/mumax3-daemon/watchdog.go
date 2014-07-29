package main

// Watchdog kills the subprocess if
// the lock directory disappears.

import (
	"log"
	"os"
	"os/exec"
	"sync"
	"time"
)

var (
	watchmutex sync.Mutex
	watchcmd   *exec.Cmd
	watchlock  string
)

// Sets the command and lock dir to be polled
// by the watchdog. If the lock dir disappears,
// cmd is killed.
func SetWatchdog(cmd *exec.Cmd, lockdir string) {
	watchmutex.Lock()
	defer watchmutex.Unlock()

	watchcmd = cmd
	watchlock = lockdir
}

func RunWatchdog() {
	log.Println("started watchdog")
	for {
		watchmutex.Lock()
		if watchlock != "" && watchcmd != nil {
			_, err := os.Stat(watchlock)
			if err != nil {
				log.Println(err)
				log.Println("killing process behind", watchlock)
				errK := watchcmd.Process.Kill()
				if errK != nil {
					log.Println(errK)
				}
				watchlock = ""
				watchcmd = nil
			}
		}
		watchmutex.Unlock()
		time.Sleep(*flag_poll)
	}
}
