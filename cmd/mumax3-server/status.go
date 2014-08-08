package main

// Serves human-readable status information over http.

import (
	"html/template"
	"log"
	"net/http"
	"time"
)

var (
	templ = template.Must(template.New("status").Parse(templText))
)

func (n *Node) HandleStatus(w http.ResponseWriter, r *http.Request) {
	log.Println("handle http status")
	err := templ.Execute(w, node.Status())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type NodeStatus struct {
	NodeInfo
	Queue  []string
	Uptime time.Duration
}

func (n *Node) Status() NodeStatus {
	n.lock()
	defer n.unlock()
	return NodeStatus{
		NodeInfo: n.inf,
		Queue:    n.jobs.ListFiles(),
		Uptime:   time.Since(n.upSince),
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

Uptime: {{.Uptime}} <br/>

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
