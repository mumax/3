/*
Easy-to-use cluster management tool for mumax3, with auto-configuration and web interface.  When nodes are connected behind a home router, mumax3-server can run without any configuration. Otherwise only the IP address range where the other nodes reside has to be specified.

# Input files

Upon starting mumax3-server, it scans the current working directory for input files. These should be organised in directories corresponding to user names. E.g.:

	john/file1.mx3
	john/file2.mx3
	...
	kate/file1.mx3
	kate/file2.mx3
	...

Other files will be ignored. These input files will run on all available nodes in the network. After adding/removing files, you should click "rescan" in the web interface, or wait for a few minutes.

# Web interface

mumax3-server serves a web interface at http://localhost:35360 (you have overridden the port, see below). Depending on your OS you may need to use your exact IP address instead of localhost, e.g.: http://192.168.0.1:35360.

The web interface shows you the queued jobs, running jobs, output files, etc., and allows to re-scan for new job files or kill running jobs.

# Compute nodes

Each node that runs mumax3-server and has a working mumax3 installation will automatically serve as a compute node (even if it stores input files as well). The web interface will show the mumax version and available GPUs. The -exec flag may be used to override which mumax3  binary to use. E.g:

	mumax3-server -exec /usr/local/mumax3/mumax3-cuda6.5  #override mumax3 binary

# Scan for other nodes

Upon starting mumax3-server, it will automatically scan for other nodes in the local network. These will automatically start running jobs (if they have a GPU and mumax3 installed), or may serve job files to be executed by other nodes.

By default, we search for nodes with IP addresses in the range 192.168.0.1-128 (local network behind, e.g., a router). This can be changed by the -scan flag. E.g.:

	mumax3-server -scan 127.0.0.1,169.254.0-1.1-254
	mumax3-server -ports 35360-25369

Even when a new node appears on the network after the port scan, it should still be automatically detected. If not, hit "rescan" in the web interface. The -ports flag may be used to change the port numbers being scanned, in case the server uses a non-standard port (-l flag).

# Override port number

mumax3-server uses tcp port 35360, which needs to be accessible (e.g., through your firewall). This port and the service's IP address, can be overridden with the -l flag:

	mumax3-server -l :35361               #serves at non-standard port
	mumax3-server -l 192.168.1.1:35360    #serves at specific IP address, e.g. for dual-link machines

# Fault tolerance

mumax3-server does a great effort to recover from failed nodes, network outages, reboots etc. If a simulation is interrupted for any such reason, it should be re-queued and automatically re-started later. In that case the web interface will show [1x requeued] to indicate that the job has been interrupted, but it will run later nevertheless.

# Command line flags

Usage of mumax3-server:

	-cache="": mumax3 kernel cache path
	-exec="mumax3": mumax3 executable
	-halflife=24h0m0s: share decay half-life
	-l=":35360": Listen and serve at this network address
	-log=true: log debug output
	-ports="35360-35361": Scan these ports for other servers
	-scan="192.168.0.1-128": Scan these IP address for other servers
	-timeout=2s: Portscan timeout

# Web interface example

http://localhost:35360

	157.193.57.146:35360

	Uptime: 27h45m38s

	Peer nodes
	 scan 157.193.57.2-254: 35360-35361
	 ports 35360-35361
	 (Rescan)
	 157.193.57.146:35360
	 157.193.57.228:35360

	Compute service
	 mumax: mumax 3.6 beta2 linux_amd64 go1.3.3 (gc)
	 GPU0: CUDA 6 GeForce GTX 680(2047MB) cc3.0
	 GPU1: CUDA 6 GeForce GTX 680(2047MB) cc3.0
	 GPU2: CUDA 6 GeForce GTX 680(2047MB) cc3.0

	Running jobs
	 [157.193.57.146:35360/john/b_ext_add.mx3]   [3s]	[GUI]	[kill]
	 [157.193.57.146:35360/john/demag2D.mx3]     [2s]	[GUI]	[kill]
	 [157.193.57.146:35360/john/demag2Dpbc.mx3]  [1s]	[GUI]	[kill]

	Queue service

	 Users
	  john    589 GPU-seconds	has queued jobs
	  kate    0   GPU-seconds	no queued jobs
	  Next job for: john

	 Jobs
	  [Reload all]
	  [Wake-up Watchdog]

	 john
	  [Reload]

	  [john/anisenergy.mx3]	                [.out]	[157.193.57.146:35360]	[ OK ]	[1s]
	  [john/anisenergyconservation.mx3]     [.out]	[157.193.57.146:35360]	[ OK ]	[2s]
	  [john/anisenergyconservation2.mx3]	[.out]	[157.193.57.146:35360]	[ OK ]	[2s]
	  [john/anisenergyconservation3.mx3]	[.out]	[157.193.57.228:35360]	[ OK ]	[1s]
	  [john/anisenergyconservation4.mx3]	[.out]	[157.193.57.146:35360]	[ OK ]	[2s]

	 kate
	 [Reload]
*/
package main
