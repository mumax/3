package main

import (
	"log"
	"time"
)

var runWatchdog = make(chan struct{})

func init() {
	// run watchdog daemon in background
	go func() {
		for {
			<-runWatchdog // wait for start
			DoWatchdog()
		}
	}()
}

func LoopWatchdog() {
	for {
		WakeupWatchdog("")
		time.Sleep(3 * KeepaliveInterval)
	}
}

func WakeupWatchdog(string) string {
	select {
	default:
		return "already running"
	case runWatchdog <- struct{}{}:
		return "" // ok
	}
}

// single watchdog run:
// re-queues all dead processes
func DoWatchdog() {
	//log.Println("Watchdog wake-up")
	WLock()
	defer WUnlock()
	for _, u := range Users {
		for _, j := range u.Jobs {
			id := j.ID
			//log.Println(id, "running:", j.IsRunning(), "alive:", time.Since(j.Alive))
			if j.IsRunning() && time.Since(j.Alive) > 3*KeepaliveInterval {
				j.Update()
				lastHeartbeat := time.Since(j.Alive)
				if lastHeartbeat > 3*KeepaliveInterval {
					log.Println("*** Re-queue", id, "after", lastHeartbeat, "inactivity")
					j.Reque()
				}
			}
		}
		// re-set nextPtr to beginning so we can start re-queued jobs
		if u.nextPtr >= len(u.Jobs) {
			u.nextPtr = 0
		}
	}
}
