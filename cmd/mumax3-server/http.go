package main

// Serves human-readable status information over http.

import (
	"html/template"
	"log"
	"net/http"
)

var (
	templ = template.Must(template.New("status").Parse(templText))
)

func ServeHTTP() {
	http.Handle("/", node)
	log.Print("Serving human-readible status at http://", *flag_http)
	Fatal(http.ListenAndServe(*flag_http, nil))
}

func (n *Node) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	err := templ.Execute(w, node.Info())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

const templText = `
	{{.Addr}}
`
