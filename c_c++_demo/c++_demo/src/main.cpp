

#include <assert.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>


#define BIND_MULTI_CORES 0

static void print_affinity() {
    cpu_set_t mask;
    long nproc, i;

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity");
        assert(false);
    }
    nproc = sysconf(_SC_NPROCESSORS_ONLN);
    printf("sched_getaffinity = ");
    for (i = 0; i < nproc; i++) {
        printf("%d ", CPU_ISSET(i, &mask));
    }
    printf("\n");
}



int main(void) {
    using namespace std;


#if BIND_MULTI_CORES
    cpu_set_t mask;

    print_affinity();
    printf("sched_getcpu = %d\n", sched_getcpu());
    // CPU_ZERO(&mask);
    // CPU_SET(0, &mask);

    CPU_SET(7, &mask);
    CPU_SET(6, &mask);
    CPU_SET(5, &mask);
    CPU_SET(4, &mask);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_setaffinity");
        assert(false);
    }
    print_affinity();
    /* TODO is it guaranteed to have taken effect already? Always worked on my tests. */
    printf("sched_getcpu = %d\n", sched_getcpu());
#endif


    cout << "Hi, .. " << endl;

    return 0;
}
