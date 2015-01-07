#! /bin/bash 

rm -rf *.out http\:
killall mumax3-httpfsd
mumax3-httpfsd -l :35377 &
sleep 1s
mumax3 $(for f in *.mx3; do echo -n ' ' http://localhost:35377/$f; done) || exit 1
killall mumax3-httpfsd
if [ -e http\: ]; then
		echo "http:" exists
		exit 2
fi;
