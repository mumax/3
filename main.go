package main

import (
	"nimble-cube/mm"
	"nimble-cube/nc"
	"log"
	"os"
)

func main() {
	log.SetPrefix("#")
	nc.PrintInfo(os.Stdout)
	mm.Main()
}
