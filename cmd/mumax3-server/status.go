package main

// Serves human-readable status information over http.

import (
	"html/template"
	"net/http"
)

var (
	templ = template.Must(template.New("status").Parse(templText))
)

func (n *Node) HandleStatus(w http.ResponseWriter, r *http.Request) {
	n.lock()
	defer n.unlock()
	err := templ.Execute(w, node)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (n *Node) IPRange() string {
	return *flag_scan + ": " + *flag_ports
}

const templText = `
<html>

<head>
	<style>
		body{font-family:monospace}
	</style>
	<meta http-equiv="refresh" content="1">
</head>

<body>

<h1>{{.Addr}}</h1>

Uptime: {{.Uptime}} <br/>

<h2>Compute service</h2>

<h3>GPUs</h3>
{{with .GPUs}}
	{{range .}}
		{{.Info}}<br/>
	{{end}}
{{else}}
	No GPUs available<br/>
{{end}}

<h3>Mumax</h3>
{{with .MumaxVersion}}
	Has {{.}}
{{else}}
	No mumax available<br/>
{{end}}

<h3>Jobs running on this node</h3>
{{range .RunningHere}}
	[GPU {{.GPU}}] [<a href="{{.File}}">{{.File}}</a>] [{{.Runtime}}] <br/>
{{end}}

<h2>Queue service</h2>
Storage root: <a href="http://{{.Addr}}/fs/{{.RootDir}}">{{.RootDir}}</a>

{{range $k,$v := .Users}}
	<h3>{{$k}}</h3>
	Share: {{.Used}}/{{.Share}} <br/>
	{{range $v.Jobs}}
		<a href="http://{{$.Addr}}/fs/{{.File}}">{{.File}}</a> {{.Node}}/{{.GPU}} <br/>
	{{end}}
{{end}}

<h2>Port scanner service</h2>
Peers in IP:port range {{.IPRange}}:<br/>
{{range .Peers}}
	<a href="http://{{.}}">{{.}}</a><br/>
{{end}}

</body>
</html>
`
