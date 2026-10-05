// wtime: wall-clock time of whole processes with minimal harness overhead.
// usage: wtime <reps> <cmd> [args...]   -> prints one line per rep: "<wall_s> <user_s> <sys_s> <maxrss_kb>"
// stdout/stderr of the child are inherited.
#define _GNU_SOURCE
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
extern char **environ;
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: wtime <reps> <cmd> [args...]\n"); return 2; }
    int reps = atoi(argv[1]);
    for (int r = 0; r < reps; r++) {
        struct timespec t0, t1;
        pid_t pid;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (posix_spawn(&pid, argv[2], NULL, NULL, argv + 2, environ) != 0) { perror("spawn"); return 1; }
        int st; struct rusage ru;
        if (wait4(pid, &st, 0, &ru) < 0) { perror("wait4"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { fprintf(stderr, "child failed (%d)\n", st); return 1; }
        double w = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        fprintf(stderr, "%.6f %.6f %.6f %ld\n", w,
                ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec,
                ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec, ru.ru_maxrss);
    }
    return 0;
}
