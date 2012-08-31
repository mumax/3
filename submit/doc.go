/*
 submit can be used to submit jobs to a mumax-daemon que.
 submit takes a flag describing the kind of job file to make
 and a number of input files. E.g.:
 	submit --mumax2 *.py
 This will generate JSON job files in the user's que directy.
 Default is $HOME/que, can be overridden with --que flag.
 The que directory should be accessible by the daemons.

 The job files are simple JSON encodings of the Job data type
 and may as well be written by hand or a script, 
 should more flexibility be needed.  E.g.:
 	{"Command":"mumax2","Args":["-gpu=%GPU","job.py"],"Wd":"/home/me/myfiles"}

 Author: Arne Vansteenkiste
*/
package main
