sed -i '/#include <math.h>/a typedef struct { uint64_t mask; double r; double i; } HopBranch;\ntypedef struct { int n0; int n1; double r; double i; } ResBranch;' /tmp/su2-254-mp/work/heis_c.c
sed -i '1,2d' /tmp/su2-254-mp/work/heis_c.c
