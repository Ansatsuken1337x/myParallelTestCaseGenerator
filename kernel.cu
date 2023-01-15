#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define TESTCASE          "Z://source//CudaRuntime1//testcases//1.jpg"
#define TMPMUTATIONSDIR   "Z://source//CudaRuntime1//mutations"
#define TARGETFORMATFILE  "jpg"
#define MAX_MUTATIONS     100
#define MAX_SAMPLES       20
#define MAX_CPU_THREADS   6
#define MUTATION_RATE     0.01
#define MAGIC_VALS_SIZE   11
#define MAX_FILENAME_SIZE 1024
#define BUFLEN            4096

#define RandomNumberInInterval(min,max) rand()%(max-min+1)+min

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

clock_t start, end;
double time_taken1=0, time_taken2=0, time_taken_gpu_avg, time_taken_cpu_avg;

// Create a structure to hold the function parameters.
struct FunctionParams
{
    unsigned char* mutatedTestcases;
    unsigned long filelen;
    float         mutation_rate;
    int  offset;
};

cudaError_t mutateTestcasesWithCuda(unsigned char *h_mutatedTestcases, unsigned long filelen, float mutation_rate);

__global__ void insertMagicKernel(unsigned char *data, unsigned long filelen, float flip_percent, unsigned int rand_magic_val_index, unsigned int picked_index);

VOID CALLBACK WorkCallback(PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work);

/* ****************************************************************************************** */


void checkGpuMem() {
    float free_m, total_m, used_m;
    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);
    free_m = (unsigned int)free_t / 1048576.0;
    total_m = (unsigned int)total_t / 1048576.0;
    used_m = total_m - free_m;
    printf("  mem free %f MB \nmem total %f MB \nmem used %f MB\n", free_m, total_m, used_m);
}

unsigned char* get_bytes(char* filename, unsigned int* filelen) {

    FILE* fileptr;
    unsigned char* data;
    int length;

    fileptr = fopen(filename, "rb");  // Open the file in binary mode
    fseek(fileptr, 0, SEEK_END);      // Jump to the end of the file
    length = ftell(fileptr);          // Get the current byte offset in the file
    rewind(fileptr);                  // Jump back to the beginning of the file

    data = (unsigned char*)malloc(length);    // Enough memory for the file
    fread(data, length, 1, fileptr);  // Read in the entire file

    if (filelen)
        *filelen = length;

    fclose(fileptr);                  // Close the file
    return data;
}

unsigned char* createNewTestCase(unsigned char* data, int filelen, int counter) {

    FILE* write_ptr;
    int totalRemaining = 0;
    int nwritten = 0;
    int blockSize = 0;
    unsigned char* newFileName = (unsigned char*)malloc(MAX_FILENAME_SIZE);

    snprintf((char*)newFileName, MAX_FILENAME_SIZE, "%s//subject-%d.%s", TMPMUTATIONSDIR, counter, TARGETFORMATFILE);

    write_ptr = fopen((char*)newFileName, "wb");
    if (write_ptr == NULL) {
        /* handle error */
        perror("file open for reading");
        exit(EXIT_FAILURE);
    }

    totalRemaining = filelen - nwritten;
    while (totalRemaining > 0) {
        blockSize = (totalRemaining >= BUFLEN) ? BUFLEN : totalRemaining;
        nwritten = fwrite(data, 1, blockSize, write_ptr);
        data += nwritten;
        totalRemaining -= nwritten;
    }

    fclose(write_ptr);
    return newFileName;
}

__global__ void insertMagicKernel(unsigned char *data, unsigned long filelen, float flip_percent, unsigned int rand_magic_val_index, unsigned int rand_picked_index) {
    unsigned int* magic_index, picked_magic[2];
    unsigned int magic_vals[11][2] = {
        {1, 255},
        {1, 255},
        {1, 127},
        {1, 0},
        {2, 255},
        {2, 0},
        {4, 255},
        {4, 0},
        {4, 128},
        {4, 64},
        {4, 127},
    };
    int num_of_flips = filelen * flip_percent;
    int flipCounter = 0;

    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within the range of the array
    if (idx >= MAX_MUTATIONS) return;

    do {
        magic_index = (unsigned int*)&magic_vals[rand_magic_val_index];
        picked_magic[0] = magic_vals[*magic_index][0];
        picked_magic[1] = magic_vals[*magic_index][1];

        // here we are hardcoding all the byte overwrites for all of the tuples that begin (1, )
        if (picked_magic[0] == 1) {
            if (picked_magic[1] == 255)			    // 0xFF
               data[idx * filelen + rand_picked_index] = 0xFF;
            else if (picked_magic[1] == 127)		// 0x7F
               data[idx * filelen + rand_picked_index] = 0x7F;
            else if (picked_magic[1] == 0)			// 0x00
               data[idx * filelen + rand_picked_index] = 0x00;
        }
        // here we are hardcoding all the byte overwrites for all of the tuples that begin (2, )
        else if (picked_magic[0] == 2) {
            if (picked_magic[1] == 255) {	      // 0xFFFF
               data[idx * filelen + rand_picked_index]     = 0xFF;
               data[idx * filelen + rand_picked_index + 1] = 0xFF;
            }
            else if (picked_magic[1] == 0) {    // 0x0000
               data[idx * filelen + rand_picked_index] = 0x00;
               data[idx * filelen + rand_picked_index + 1] = 0x00;
            }
        }
        // here we are hardcoding all of the byte overwrites for all of the tuples that being (4, )
        else if (picked_magic[0] == 4) {
            if (picked_magic[1] == 255) {  // 0xFFFFFFFF
               data[idx * filelen + rand_picked_index] = 0xFF;
               data[idx * filelen + rand_picked_index + 1] = 0xFF;
               data[idx * filelen + rand_picked_index + 2] = 0xFF;
               data[idx * filelen + rand_picked_index + 3] = 0xFF;
            }
            else if (picked_magic[1] == 0) {    // 0x00000000
               data[idx * filelen + rand_picked_index] = 0x00;
               data[idx * filelen + rand_picked_index + 1] = 0x00;
               data[idx * filelen + rand_picked_index + 2] = 0x00;
               data[idx * filelen + rand_picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 128) {  // 0x80000000
               data[idx * filelen + rand_picked_index] = 0x80;
               data[idx * filelen + rand_picked_index + 1] = 0x00;
               data[idx * filelen + rand_picked_index + 2] = 0x00;
               data[idx * filelen + rand_picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 64) {   // 0x40000000
               data[idx * filelen + rand_picked_index] = 0x40;
               data[idx * filelen + rand_picked_index + 1] = 0x00;
               data[idx * filelen + rand_picked_index + 2] = 0x00;
               data[idx * filelen + rand_picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 127) {  // 0x7FFFFFFF
               data[idx * filelen + rand_picked_index] = 0x7F;
               data[idx * filelen + rand_picked_index + 1] = 0xFF;
               data[idx * filelen + rand_picked_index + 2] = 0xFF;
               data[idx * filelen + rand_picked_index + 3] = 0xFF;
            }
        }
        flipCounter++;
    } while (flipCounter < num_of_flips);

    return;
}

void insert_magic(unsigned char* data, long filelen, float mutatation_rate) {
    unsigned int* magic_index, picked_magic[2], picked_index;
    unsigned int magic_vals[11][2] = {
        {1, 255},
        {1, 255},
        {1, 127},
        {1, 0},
        {2, 255},
        {2, 0},
        {4, 255},
        {4, 0},
        {4, 128},
        {4, 64},
        {4, 127},
    };
    int num_of_flips = filelen * mutatation_rate;
    int flipCounter = 0;

    do
    {
        magic_index = (unsigned int*)&magic_vals[rand() % 11];
        picked_magic[0] = magic_vals[*magic_index][0];
        picked_magic[1] = magic_vals[*magic_index][1];
        picked_index = RandomNumberInInterval(6, filelen);

        // here we are hardcoding all the byte overwrites for all of the tuples that begin (1, )
        if (picked_magic[0] == 1) {
            if (picked_magic[1] == 255)			    // 0xFF
                data[picked_index] = 0xFF;
            else if (picked_magic[1] == 127)		// 0x7F
                data[picked_index] = 0x7F;
            else if (picked_magic[1] == 0)			// 0x00
                data[picked_index] = 0x00;
        }
        // here we are hardcoding all the byte overwrites for all of the tuples that begin (2, )
        else if (picked_magic[0] == 2) {
            if (picked_magic[1] == 255) {	      // 0xFFFF
                data[picked_index] = 0xFF;
                data[picked_index + 1] = 0xFF;
            }
            else if (picked_magic[1] == 0) {    // 0x0000
                data[picked_index] = 0x00;
                data[picked_index + 1] = 0x00;
            }
        }
        // here we are hardcoding all of the byte overwrites for all of the tuples that being (4, )
        else if (picked_magic[0] == 4) {
            if (picked_magic[1] == 255) {  // 0xFFFFFFFF
                data[picked_index] = 0xFF;
                data[picked_index + 1] = 0xFF;
                data[picked_index + 2] = 0xFF;
                data[picked_index + 3] = 0xFF;
            }
            else if (picked_magic[1] == 0) {    // 0x00000000
                data[picked_index] = 0x00;
                data[picked_index + 1] = 0x00;
                data[picked_index + 2] = 0x00;
                data[picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 128) {  // 0x80000000
                data[picked_index] = 0x80;
                data[picked_index + 1] = 0x00;
                data[picked_index + 2] = 0x00;
                data[picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 64) {   // 0x40000000
                data[picked_index] = 0x40;
                data[picked_index + 1] = 0x00;
                data[picked_index + 2] = 0x00;
                data[picked_index + 3] = 0x00;
            }
            else if (picked_magic[1] == 127) {  // 0x7FFFFFFF
                data[picked_index] = 0x7F;
                data[picked_index + 1] = 0xFF;
                data[picked_index + 2] = 0xFF;
                data[picked_index + 3] = 0xFF;
            }
        }
        flipCounter++;
    } while (flipCounter < num_of_flips);

    return;
}

void WorkCallback(PTP_CALLBACK_INSTANCE instance, PVOID param, PTP_WORK work)
{    
    FunctionParams *localstruct = (FunctionParams*)param;
    unsigned long  filelen = localstruct->filelen;
    unsigned int   offset  = localstruct->offset;

    unsigned char* testcaseToMutate = (unsigned char*)malloc(filelen);
    memcpy(testcaseToMutate, localstruct->mutatedTestcases + (filelen * offset), filelen);

    insert_magic(testcaseToMutate, filelen, MUTATION_RATE);

    // Write testcase to file
    //unsigned char* mutationFileName = createNewTestCase(testcaseToMutate, filelen, 0);
    
    // This function will be executed in a worker thread from the thread pool.
    //printf("Hello from the worker thread!\n");

    //free(testcaseToMutate);
}

void __multithread_mutate_testcases(unsigned char* mutatedTestcases, unsigned long filelen, float mutation_rate) {

    PTP_WORK       work;
    FunctionParams params;

    // Allocate memory for the structure.
    unsigned long size2DArrayTestcases = MAX_MUTATIONS * filelen;
    // Set the values of the structure.
    params.mutatedTestcases = (unsigned char*)malloc(size2DArrayTestcases);
    memcpy(params.mutatedTestcases, mutatedTestcases, size2DArrayTestcases);
    params.filelen       = filelen;
    params.mutation_rate = mutation_rate;
    params.offset        = -1;

    //work = CreateThreadpoolWork(WorkCallback, &params, NULL);
    //if (work)
    //{
    //    
    //    for (size_t i = 0; i < MAX_SAMPLES; i++)
    //    {
    //        start = clock();

    //        for (size_t j = 0; j < MAX_MUTATIONS; j++)
    //        {
    //            ++params.offset;

    //            // Submit the work item to the thread pool.
    //            SubmitThreadpoolWork(work);
    //        }

    //        // Wait for the work item to complete.
    //        WaitForThreadpoolWorkCallbacks(work, FALSE);

    //        end = clock();
    //        // Tempo total SEM aceleracao
    //        time_taken2 += double(end - start) / double(CLOCKS_PER_SEC);
    //    }
    //    
    //    // Clean up the work item when you are finished with it.
    //    CloseThreadpoolWork(work);
    //}

    
    PTP_POOL pool = CreateThreadpool(NULL);
    if (pool != NULL)
    {
        SetThreadpoolThreadMinimum(pool, MAX_CPU_THREADS);
        work = CreateThreadpoolWork(WorkCallback, &params, NULL);

        if (work != NULL)
        {
            for (size_t i = 0; i < MAX_SAMPLES; i++)
            {
                params.offset = -1;
                start = clock();

                for (size_t j = 0; j < MAX_MUTATIONS; j++)
                {
                    ++params.offset;

                    // Submit the work item to the thread pool.
                    SubmitThreadpoolWork(work);
                }

                WaitForThreadpoolWorkCallbacks(work, FALSE);

                end = clock();
                // Tempo total SEM aceleracao
                time_taken2 += double(end - start) / double(CLOCKS_PER_SEC);
            }            

            CloseThreadpoolWork(work);
        }
        CloseThreadpool(pool);
    }


    time_taken_cpu_avg = time_taken2 / MAX_SAMPLES;
    printf("\n(T2) Tempo medio SEM aceleracao na placa: %lf\n", time_taken_cpu_avg);
}

int main() 
{
    cudaError_t cudaStatus;
    unsigned char* testcaseFileData;
    unsigned int   filelen;    
    srand((unsigned)time(NULL)); // Initialize the random seed

    // Check available video card memory;
    //checkGpuMem();

    // Read the testcase
    testcaseFileData = get_bytes(TESTCASE, &filelen);

    // Write testcase to file
    // unsigned char *mutationFileName = createNewTestCase(testcaseFileData, filelen, 0);

    // Allocate device arrays
    unsigned char* h_mutatedTestcases;
    int len = sizeof(unsigned char) * MAX_MUTATIONS * filelen;
    h_mutatedTestcases = (unsigned char*)malloc(len);

    for (size_t i = 0; i < MAX_MUTATIONS; i++)
    {
        for (size_t j = 0; j < filelen; j++)
        {
            h_mutatedTestcases[i * filelen + j] = testcaseFileData[j];
        }
    }

    // Obtem o tamanho maximo do grid, do block e de threads per block
    int devNo = 0;
    cudaDeviceProp iProp; cudaGetDeviceProperties(&iProp, devNo); printf("Maximum grid size is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxGridSize[i]); printf(")\n"); printf("Maximum block dim is: (");
    for (int i = 0; i < 3; i++)
        printf("%d\t", iProp.maxThreadsDim[i]); printf(")\n"); printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);


    /* ******************** COM ACELERACAO ******************** */
    // Mutate testcase in parallel.
    cudaStatus = mutateTestcasesWithCuda(h_mutatedTestcases, filelen, MUTATION_RATE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mutateTestcasesWithCuda failed!\n");
        fprintf(stderr, "Error: %s\n\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    /* ******************** SEM ACELERACAO ******************** */
    __multithread_mutate_testcases(h_mutatedTestcases, filelen, MUTATION_RATE);



    return 0;
}

// Helper function for mutate testcases in parallel.
cudaError_t mutateTestcasesWithCuda(unsigned char * h_mutatedTestcases, unsigned long filelen, float mutation_rate)
{
    cudaError_t   cudaStatus;
    unsigned int  rand_picked_index    = RandomNumberInInterval(6, filelen-1);
    unsigned int  rand_magic_val_index = rand() % MAGIC_VALS_SIZE;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate device arrays
    unsigned char* d_mutatedTestcases;
    CHECK(cudaMalloc((void**)&d_mutatedTestcases, filelen * MAX_MUTATIONS));
    //
    //// Copy data from host to device
    //CHECK(cudaMemcpy(d_mutatedTestcases, h_mutatedTestcases, filelen * MAX_MUTATIONS, cudaMemcpyHostToDevice));
    

    // The total number of threads in a block is a block
    // The total number of blocks in a grid is grid
    /*
    int block1 = 32;
    int grid1 = (MAX_MUTATIONS + block1 - 1) / block1;*/

    int nx;    //total threads in X dimension
    int ny;    //total threads in Y dimension
    int nz;    //total threads in Z dimension

    nx = 128;       //128 threads in X dim
    ny = nz = 1;    //1 thread in Y & Z dim

    //32 threads in X and 1 each in Y & Z in a block
    dim3 block2(32, 1, 1); //4 blocks in X & 1 each in Y & Z
    dim3 grid2(nx / block2.x, ny / block2.y, nz / block2.z);

    for (size_t i = 0; i < MAX_SAMPLES; i++)
    {
        // Launch the kernel to manipulate the strings in parallel
        start = clock();

          // Copy data from host to device
        CHECK(cudaMemcpy(d_mutatedTestcases, h_mutatedTestcases, filelen * MAX_MUTATIONS, cudaMemcpyHostToDevice));

        //insertMagicKernel <<< grid1, block1 >>> (d_mutatedTestcases, filelen, MUTATION_RATE, rand_magic_val_index, rand_picked_index);
        insertMagicKernel <<< grid2, block2 >>> (d_mutatedTestcases, filelen, MUTATION_RATE, rand_magic_val_index, rand_picked_index);


        // Copy data from device to host
        CHECK(cudaMemcpy(h_mutatedTestcases, d_mutatedTestcases, filelen * MAX_MUTATIONS, cudaMemcpyDeviceToHost));

        end = clock();
        // Tempo total COM aceleracao na GPU
        time_taken1 += double(end - start) / double(CLOCKS_PER_SEC);
    }
    

    // Write testcase to file
    //unsigned char *mutationFileName = createNewTestCase(h_mutatedTestcases, filelen, 0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "insertMagicKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching insertMagicKernel!\n", cudaStatus);
    }



    time_taken_gpu_avg = time_taken1 / MAX_SAMPLES;
    printf("\n(T1) Tempo medio COM aceleracao na placa: %lf", time_taken_gpu_avg);

    cudaFree(d_mutatedTestcases);
    return cudaStatus;
}