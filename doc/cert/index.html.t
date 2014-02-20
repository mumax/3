{{.Inc "header"}}

<h1>Mumax certification</h1>

Mumax solves

{{.Inc "formula" "llg" }}

with variables accessible as:<br/>
{{.Inc "tex" `\vec m`}} = {{.Inc "api" "m" }} <br/>
{{.Inc "tex" `\gamma_\mathrm{LL}`}} = {{.Inc "api" "gammaLL" }} <br/>
{{.Inc "tex" `\vec\tau` }}          = {{.Inc "api" "torque"}} 

<h1>torque</h1>

Landau-Lifshitz torque expressed in Tesla:

{{.Inc "formula" "lltorque"}}

with variables accessible as:<br/>
{{.Inc "tex" `\alpha`}}                  {{.Inc "api" "alpha" }} <br/>
{{.Inc "tex" `\vec{B}_\mathrm{eff}` }}   {{.Inc "api" "B_eff"}} = {{.Inc "api" "B_exch" }} + {{.Inc "api" "B_demag" }} + {{.Inc "api" "B_anis" }} + {{.Inc "api" "B_ext" }}

<h2>LLtorque test</h2>

Damping-less precession of single spin in 100mT field:
{{.Inc "input" "precession.txt"}}
{{.Inc "img"   "precession.svg"}}

<h1>exchange</h1>

The exchange field is defined as:

{{.Inc "formula" "bexch" }}

with variables accessible as:<br/>
{{.Inc "tex" `A`}} = {{.Inc "api" "Aex" }} <br/>
{{.Inc "tex" `D`}} = {{.Inc "api" "Dex" }} <br/>

<p>The first term is the Heisenberg exchange. The discretized form uses a 6-neighbor approximation with Neumann boundary conditions. See M.J. Donahue and D.G. Porter, Physica B, 343, 177-183 (2004).</p>



<p>The second term the Dzyaloshinskii-Moriya interaction according to Bagdanov and Röβler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).</p>



Mumax uses is a linear approximation suitable for small spin-spin angles. It is the user's responsibility to choose a sufficiently small cell size.
We test the exchange interaction by setting the magnetization to a uniform spiral and calculating the exchange energy as a function of the spiral period.

{{.Inc "input" "exchange1d.txt"}}
{{.Inc "img"   "exchange1dspiral.svg"}}
{{.Inc "img"   "exchange1d.svg"}}
{{.Inc "img"   "exchange1d2.svg"}}


<h1>Solver precission</h1>
{{.Inc "input" "heun.txt"}}
{{.Inc "img"   "heun1.svg"}}
{{.Inc "img"   "heun2.svg"}}



{{.Inc "footer"}}


