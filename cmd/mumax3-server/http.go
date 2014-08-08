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
	err := templ.Execute(w, node.Status())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type NodeStatus struct {
	NodeInfo
	Queue []string
}

func (n *Node) Status() NodeStatus {
	n.lock()
	defer n.unlock()
	return NodeStatus{
		NodeInfo: n.inf,
		Queue:    n.jobs.ListFiles(),
	}
}

const templText = `
<html>

<head>
	<style>
		body{font-family:monospace}
	</style>
</head>

<body>

<h1>{{.Addr}}</h1>

{{with .MumaxVersion}}
	Running {{.}}
{{end}}

{{with .GPUs}}
	<h2>GPUs</h2>
	{{range .}}
		{{.Info}}</br>
	{{end}}
{{end}}

<h2>Queue</h2>
{{range .Queue}}
	{{.}}</br>
{{end}}

</body>
</html>
`
