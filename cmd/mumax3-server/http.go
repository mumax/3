package main

// Serves human-readable status information over http.

import (
	"encoding/json"
	"log"
	"net/http"
)

func ServeHTTP() {
	http.HandleFunc("/", handleHTTP)
	log.Print("Serving human-readible status at http://", *flag_http)
	Fatal(http.ListenAndServe(*flag_http, nil))
}

func handleHTTP(w http.ResponseWriter, r *http.Request) {
	var info map[string]interface{}
	PipeAndWait(func() {
		info = map[string]interface{}{
			"Addr":  MyStatus.Addr,
			"Type":  MyStatus.Type,
			"Peers": peers,
			"Jobs":  jobs.ListFiles(),
		}
	})
	bytes, err := json.MarshalIndent(info, "", "\t")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
	w.Write(bytes)
}
