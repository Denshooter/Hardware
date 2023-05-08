#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char *argv[])
{
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello world. %d\n", thread_id);
    }
    return 0;
}
