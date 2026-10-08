# Move hash_func to top
sed -i '/static inline uint32_t hash_func/,+8d' /tmp/su2-254-mp/work/heis_c.c
sed -i '/void clear_hash/i static inline uint32_t hash_func(uint64_t c0, uint64_t a0, uint64_t c1, uint64_t a1) {\n    uint64_t h = c0 ^ (a0 * 11) ^ (c1 * 131) ^ (a1 * 1313);\n    h ^= h >> 33;\n    h *= 0xff51afd7ed558ccd;\n    h ^= h >> 33;\n    h *= 0xc4ceb9fe1a85ec53;\n    h ^= h >> 33;\n    return h & HASH_SIZE;\n}' /tmp/su2-254-mp/work/heis_c.c
gcc -shared -fPIC -O3 /tmp/su2-254-mp/work/heis_c.c -o /tmp/su2-254-mp/work/heis_c.so
