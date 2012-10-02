#! /bin/bash
go build -v test4.go || exit 3
export GOMAXPROCS=4
bench=test4.txt
echo "# N0 N1 N2 maxblock µs/step" > $bench
tail -f $bench &
for (( N=16; $N<2048; N+=16 )); do
	./test4 -maxblock $(( $N*$N/4 )) -log=0 1 $N $N >> $bench || exit 2
done
echo "#done" > $bench
