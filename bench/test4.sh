#! /bin/bash
go build -v test4.go || exit 3
export GOMAXPROCS=4
bench=test4.txt
echo "# N0 N1 N2 maxblock µs/step" > $bench
tail -f $bench &
delta=64
for (( N=$delta; $N<2048; N+=$delta )); do
	./test4 -minblocks 4 -silent 1 $N $N >> $bench || exit 2
done
echo "#done" >> $bench
