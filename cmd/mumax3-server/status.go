package main

// Serves human-readable status information over http.

import (
	"html/template"
	"net/http"
	"time"
)

var (
	templ = template.Must(template.New("status").Parse(templText))
)

func (n *Node) HandleStatus(w http.ResponseWriter, r *http.Request) {
	err := templ.Execute(w, node.Status())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type NodeStatus struct {
	Addr           string
	MumaxVersion   string // which mumax version this node runs, if any
	GPUs           []GPU  // number of available GPUs
	Queue, Running []Job
	Peers          []string
	Uptime         time.Duration
}

func (n *Node) Status() NodeStatus {
	n.lock()
	defer n.unlock()
	peers := make([]string, 0, len(n.peers))
	for k, _ := range n.peers {
		peers = append(peers, k)
	}
	return NodeStatus{
		Addr:    n.Addr,
		Queue:   copyJobs(n.jobs),
		Running: copyJobs(n.running),
		Uptime:  time.Since(n.upSince),
		Peers:   peers,
		GPUs:    n.GPUs, // read-only
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


<h2>Running</h2>
{{range .Running}}
	{{.File}} on {{.Node}} </br>
{{end}}

<h2>Queued</h2>
{{range .Queue}}
	{{.File}}</br>
{{end}}

<h2>Peers</h2>
{{range .Peers}}
	{{.}}</br>
{{end}}

</body>
</html>
`
