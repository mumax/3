package main

import (
	"log"
	"time"
)

func RunWatchdog() {
	for {

		for _, u := range Users {
			for id, j := range u.Jobs {
				if j.IsRunning() {
					j.Update()
					lastHeartbeat := time.Now().Sub(j.Alive)
					if lastHeartbeat > 3*KeepaliveInterval {
						log.Println("job", id, "died, re-queueing")
						j.Reque()
					}
				}
			}
		}
	}
}
