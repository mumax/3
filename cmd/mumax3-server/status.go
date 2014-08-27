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

{{define "Job"}}
	<tr>
		<td> [{{.Status.String}}] </td>
		<td> [<a href="{{.File}}">{{.File}}</a>] </td>
		<td> [{{with .Status}}GPU{{$.GPU}}{{end}}] </td>
		<td> [{{with .Status}}<a href="{{$.OutDir}}">output</a>{{end}}] </td>
		<td> [{{with .Status}}{{$.Runtime}}{{end}}] </td>
	</tr>
{{end}}

<html>

<head>
	<style>
		body{font-family:monospace; margin-left:5%; margin-top:1em}
		p{margin-left: 2em}
		h3{margin-left: 2em}
		a{text-decoration: none; color:blue}
		a:visited{text-decoration: none; color:blue}
		a:hover{text-decoration: underline; color:blue}
	</style>
	<meta http-equiv="refresh" content="1">
</head>

<body>

<h1>{{.Addr}}</h1>

Uptime: {{.Uptime}} <br/>

<h2>Compute service</h2><p>

<b>mumax:</b>
{{with .MumaxVersion}}
	{{.}}
{{else}}
	not available<br/>
{{end}}
<br/>

{{with .GPUs}}
	{{range $i, $v := .}}
		<b>GPU{{$i}}</b>: {{$v.Info}}<br/>
	{{end}}
{{else}}
	No GPUs available<br/>
{{end}}
</p>


<h3>Running jobs</h3><p>
	<table>
		{{range .RunningHere}}
			{{template "Job" .}}
		{{end}}
	</table>
</p>

<h2>Queue service</h2><p>
Storage root: <a href="http://{{.Addr}}/fs/">{{.RootDir}}</a>

{{range $k,$v := .Users}}
	<h3>{{$k}}</h3><p>
	Share: {{.Used}}/{{.Share}} <br/>
	<table> {{range $v.Running}}  {{template "Job" .}} {{end}} 
	        {{range $v.Queue}}    {{template "Job" .}} {{end}} 
	        {{range $v.Finished}} {{template "Job" .}} {{end}} </table>
	</p>
{{end}}
</p>

<h2>Port scanner service</h2>
Peers in IP:port range {{.IPRange}}:<br/>
{{range .Peers}}
	<a href="http://{{.Addr}}">{{.Addr}}</a><br/>
{{end}}

</body>
</html>
`
