/*
 daemon runs on the cluster's compute nodes. E.g.:
 	daemon --gpu=0 /home/alice/que /home/bob/que ...
 	daemon --gpu=1 /home/alice/que /home/bob/que ...
 The daemons look in the specified que directories for job files
 with .json extension. Job files are typically created with
 the "submit" program.
  
 When a job is started, a corresponding 
 .out directory is generated with some info. E.g.:
 	job.json
 	job_hostname_gpu1.out/
 		daemon.log // contains the daemon's output
 		stdout     // contains the job subprocess output

 Author: Arne Vansteenkiste
*/
package main
