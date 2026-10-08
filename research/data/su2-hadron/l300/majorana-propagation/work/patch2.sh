sed -i 's/memset(hash_heads, 0, (HASH_SIZE + 1) \* sizeof(uint32_t));/for (uint32_t i = 0; i < hash_count; i++) { uint32_t h = hash_func(hash_table[i].c0, hash_table[i].a0, hash_table[i].c1, hash_table[i].a1); hash_heads[h] = 0; }/' /tmp/su2-254-mp/work/heis_c.c
gcc -shared -fPIC -O3 /tmp/su2-254-mp/work/heis_c.c -o /tmp/su2-254-mp/work/heis_c.so
