package main

// Serves human-readable status information over http.

import (
	"html/template"
	"net/http"
	"time"
)

var (
	templ   = template.Must(template.New("status").Parse(templText))
	upSince = time.Now()
)

func HandleStatus(w http.ResponseWriter, r *http.Request) {
	RLock()
	defer RUnLock()

	if r.URL.Path != "/" {
		http.Error(w, "Does not compute", http.StatusNotFound)
		return
	}

	err := templ.Execute(w, &status{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

type status struct{} // dummy type to define template methods on

func (*status) IPRange() string              { return *flag_scan + ": " + *flag_ports }
func (*status) ThisAddr() string             { return thisAddr }
func (*status) Uptime() time.Duration        { return Since(time.Now(), upSince) }
func (*status) MumaxVersion() string         { return MumaxVersion }
func (*status) GPUs() []string               { return GPUs }
func (*status) RunningHere() map[string]*Job { return RunningHere }
func (*status) Users() map[string]*User      { return Users }
func (*status) NextUser() string             { return nextUser() }
func (*status) Peers() map[string]*Peer      { return peers }

const templText = `

{{define "Job"}}
<tr>
<td class={{.Status.String}}> [{{.Status.String}}] </td>
		<td> [<a href="{{.FS .URL}}">{{.LocalPath}}</a>] </td>
		<td> [{{with .Engaged}}<a href="http://{{.}}">{{.}}</a>{{end}}] </td>
		<td> [{{with .Output}}<a href="{{$.FS $.OutputURL}}">.out</a>{{end}}] </td>
		<td> [{{with .Status}}<a href="http://{{$.Node}}">{{$.Node}}</a>/{{$.GPU}}{{end}}] </td>
		<td> [{{with .Status}}<a href="{{$.OutputURL}}">out</a>{{end}}] </td>
		<td> [{{with .IsRunning}}<a href="http://{{$.NodeName}}:{{$.GUIPort}}">gui</a>{{end}}] </td>
		<td> [{{with .Status}}{{$.Runtime}}{{end}}] </td>
		<td> 
			{{with .Cmd}} [<a href="http://{{$.Node}}/do/kill/{{$.Path}}">kill</a>]   {{end}} 
			{{with .Reque}} [{{.}}x requeued]   {{end}} 
		</td>
	</tr>
{{end}}

<html>

<head>
	<style>
		body{font-family:monospace; margin-left:5%; margin-top:1em}
		p{margin-left: 2em}
		h3{margin-left: 2em}
		a{text-decoration: none; color:#0000AA}
		a:hover{text-decoration: underline}
		.FAILED{color:red; font-weight:bold}
		.RUNNING{font-weight: bold; color:blue}
		.QUEUED{color:black}
		.FINISHED{color: grey}
	</style>
	<meta http-equiv="refresh" content="60">
</head>

<body>

<h1>{{.ThisAddr}}</h1>

Uptime: {{.Uptime}} <br/>

<h2>Peer nodes</h2>

	{{range .Peers}} 
		{{.}} <br/>
	{{end}}

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
			<b>GPU{{$i}}</b>: {{$v}}<br/>
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

	<h3>Users</h3>
	{{range $k,$v := .Users}}
		{{$k}} {{$v.FairShare}} GPU-hour, {{with .HasJob}} has {{else}} no {{end}} queued jobs<br/>
	{{end}}
	<b>Next job for:</b> {{.NextUser}}

	<h3>Jobs</h3>
	{{range $k,$v := .Users}}
		<a id="{{$k}}"></a><h3>{{$k}}</h3><p>
		<b>Jobs:</b>
		<table> {{range $v.Jobs}} {{template "Job" .}} {{end}} </table>
		</p>
	{{end}}
	</p>
	



<h2>HTTPFS service</h2><p>
	:-)
</p>


</body>
</html>
`
