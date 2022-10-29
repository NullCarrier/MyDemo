#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifdef SEMAPHORE
#include <semaphore.h>
#include <unistd.h>

//Share var between threads
size_t inode = 0;

sem_t mutex;
  
void* thread(void* arg)
{
    //wait
    sem_wait(&mutex);
    printf("\nEntered critial section..\n");
  
    //critical section
    //sleep(4);
    printf("The inode number: %ld\n", ++inode);
      
    //signal
    printf("\nJust Exiting critical section...\n");
    sem_post(&mutex);
}
  
  
int main()
{
    //binary semaphore and shared between threads only
    sem_init(&mutex, 0, 1);
    pthread_t t1,t2;
    pthread_create(&t1,NULL,thread,NULL);
    sleep(2);
    pthread_create(&t2,NULL,thread,NULL);
    pthread_join(t1,NULL);
    pthread_join(t2,NULL);
    sem_destroy(&mutex);
    return 0;
}
#endif 

char **ptr;
void *thread(void *data);

void *thread(void *data)
{
    int myid = (int *)data;
    static int cnt = 0;
    printf("[%d]: %s (cnt=%d)\n", myid, ptr[myid], ++cnt);
    return NULL;
}


int main()
{
    pthread_t tid;
    //Create a new thread

    char *msgs[2] = {
        "Hellow from foo",
        "Hellow from bar"
    };
    ptr = msgs;

    for (int i = 0; i < 2; i++){
        pthread_create(&tid, NULL, thread, (void *) i);
        printf("The thread id, printed from main thread: %ld\n", tid);
    }

    //Wait for the thread to terminate
    pthread_exit(NULL);

    //exit(0);
}