/*
This directory contains example input scripts.


Geometry

See file examples/geometry.txt

		Geometry example: program a complex shape:
		a square with two round holes, like cheese.
		Also, the cheese rotates in time.


	setgridsize(256, 256, 1)
	setcellsize(1e-9, 1e-9, 1e-9)

	d      := 200e-9
	square := rect(d, d)                 // square with side d

	h     := 50e-9
	hole  := cylinder(h, h)              // circle with diameter h
	hole1 := hole.transl(100e-9, 0, 0)  // translated circle #1
	hole2 := hole.transl(0, -50e-9, 0)  // translated cricle #2
	cheese:= square.sub(hole1).sub(hole2)   // subtract the circles form the square (makes holes).
	setgeom(cheese)

	msat = 600e3
	aex = 12e-13
	alpha = 3

	// rotate the cheese.
	for i:=0; i<=180; i++{
		angle := i*pi/180
		setgeom(cheese.rotz(angle))
		m = uniform(cos(angle), sin(angle), 0)
		run(0.1e-9)
	}



Pma-racetrack

See file examples/pma-racetrack.txt
	// In this example we drive a domain wall in PMA material by spin-transfer torque.
	// We set up a post-step function that makes the simulation box "follow" the domain
	// wall. Like this, only a small number of cells is needed to simulate an infinitely
	// long magnetic wire.


	// Geometry
		setgridsize(128, 256, 1)
		setcellsize(2e-9, 2e-9, 1e-9)

	// Material parameters
		msat    = 600e3
		aex     = 10e-12
		alpha   = 0.02
		anisU   = vector(0, 0, 1)
		Ku1     = 0.59e6
		xi      = 0.2
		spinpol = 0.5

	// Initial magnetization
		m     = twoDomain(0, 0, 1, 1, 1, 0, 0, 0, -1) // up-down domains with wall between Bloch and Néél type
		alpha = 3                                     // high damping for fast relax
		run(0.1e-9)                                   // relax
		alpha = 0.02                                  // restore normal damping

	// Set post-step function that centers simulation window on domain wall.
		postStep(centerPMAwall)

	// Schedule output
		autosave(m, 100e-12)
		savetable(10e-12)

	// Run for 1ns with current through the sample
		j = vector(1e13, 0, 0)
		run(1e-9)


Py-racetrack

See file examples/py-racetrack.txt
	// In this example we drive a vortex wall in Permalloy by spin-transfer torque.
	// We set up a post-step function that makes the simulation box "follow" the domain
	// wall. By removing surface charges at the left and right ends, we mimic an infintely
	// long wire.

	// Geometry
		setgridsize(256, 64, 1)
		setcellsize(3e-9, 3e-9, 30e-9)

	// Material
		Msat    = 860e3
		Aex     = 13e-12
		Xi      = 0.1
		SpinPol = 0.56

	// Initial magnetization close to vortex wall
		m = vortexwall(1, -1, 1, 1)

	// Remove surface charges from left (mx=1) and right (mx=-1) sides to mimic infinitely long wire.
		RemoveLRSurfaceCharge(1, -1)

	// Set post-step function that centers simulation window on domain wall.
		PostStep(centerInplaneWall)

	// Relax
		Alpha = 3    // high damping for fast relax
		Run(1e-9)    // relax
		Alpha = 0.02 // restore normal damping

	// Schedule output
		Autosave(m, 100e-12)
		savetable(10e-12)

	// Run the simulation with current through the sample
		J = vector(-8e12, 0, 0)
		Run(1e-9)


Standardproblem4

See file examples/standardproblem4.txt
	// Micromagnetic standard problem 4 (a) according to
	// http://www.ctcms.nist.gov/~rdm/mumag.org.html

	// geometry
		setgridsize(128, 32, 1)
		setcellsize(500e-9/128, 125e-9/32, 3e-9)

	// material
		alpha = 3
		msat  = 800e3
		aex   = 13e-12
		m     = uniform(1, .1, 0)

	// output
		savetable(10e-12)
		autosave(m, 50e-12)

	// relax
		run(10e-9)
		print("relaxed m:", average(m))

	// run
		alpha = 0.02
		b_ext = vector(-24.6E-3, 4.3E-3, 0)
		run(1e-9)

		print("final m:")
		print(average(m))



Vortex

See file examples/vortex.txt
	// Switch magnetic vortex with rotating field.

	// geometry
		d := 400e-9                 // disk diameter
		h := 40e-9                  // disk thickness
		N := 128                    // number of cells
		setgridsize(N, N, 1)
		setcellsize(d/N, d/N, h)
		setgeom(cylinder(d, d))

	// material
		msat  = 800e3
		aex   = 13e-12
		m     = vortex(1, 1)  // circulation, polarization

		save(m) // to see initial state

	// relax
		alpha = 3
		run(1e-9)
		alpha = 0.01

	// output
		savetable(10e-12)
		autosave(m, 50e-12)

	// run
		f := 720e6 // excitation frequency
		A := 1e-3  // excitation amplitude
		b_ext = vector(A*cos(2*pi*f*t), A*sin(2*pi*f*t), 0)
		run(10e-9)


*/
package examples
