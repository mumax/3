#ifndef _EXCHANGE_H_
#define _EXCHANGE_H_

// indexing in symmetric matrix
#define symidx(i, j) ( (j<=i)? ( (((i)*((i)+1)) /2 )+(j) )  :  ( (((j)*((j)+1)) /2 )+(i) ) )

#endif
