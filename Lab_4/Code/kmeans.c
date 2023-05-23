#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#define CLUSTER_FILE "clusters.csv" // read data from this file
#define MEANS_FILE "means.csv"      // save computed means to this file
#define VALUES 750                  // number of data points in CLUSTER_FILE
#define K 4                         // number of means to be computed
#define EPISODES 1000000            // number of episodes to run k-means

// number of bytes in the simd vector
#define SIMD_BYTES 32
// number of floats in a simd vector
#define SIMD_FLOATS (SIMD_BYTES / sizeof(float))

/**
 * @brief   Parses a CSV-file into x and y coordinates.
 */
void readCSV(const char *filename, float *pointsx, float *pointsy, int values) {
    FILE *f = fopen(filename, "r");
    for (int i = 0; i < values; i++) {
        int ret = fscanf(f, "%f, %f", &pointsx[i], &pointsy[i]);
        if (ret != 2) {
            break;
        }
    }
    fclose(f);
}

/**
 * @brief   Writes x and y coordinates to a CSV-file.
 */
void writeCSV(const char *filename, float *pointsx, float *pointsy, int k) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < k; i++) {
        fprintf(f, "%0.3f, %0.3f\n", pointsx[i], pointsy[i]);
    }
    fclose(f);
}

/**
 * @brief   Prints a SIMD vector of floats.
 */
void printVectorf(__m256 v) {
    for (int i = 0; i < SIMD_FLOATS; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

/**
 * @brief   Prints a SIMD vector of 32 bit integers.
 */
void printVectori(__m256i v) {
    for (int i = 0; i < SIMD_FLOATS; i++) {
        // A vector of type __m256i is indexed as 64 bit values.
        // i.e. v[0] is the first 64 bit integer.
        // Since we're dealing with 32 bit values, we need to split them manually.
        long long int tmp = v[i / 2];
        int val = i & 1 ? (int)(tmp >> 32):(int)(tmp & 0xFFFFFFFF);
        printf("0x%X ", val); // print as hex 
    }
    printf("\n");
}

/**
 * @brief   Computes the distance between two points 'a' and 'b'.
 */
static inline float distance(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    return dx * dx + dy * dy;
}

/**
 * @brief       K-means algorithm.
 *
 * @param[in]   pointsx     x coordinates of input points
 * @param[in]   pointsy     y coordinates of input points
 * @param[in]   values      number of input point
 * @param[out]  meansx      x coordinates of the resulting means
 * @param[out]  meansy      y coordinates of the resulting means
 * @param[in]   k           number of means to be computed
 *
 * @return      Returns 0 for success, -1 otherwise.
 */
int kmeans(float *pointsx, float *pointsy, int values, float *meansx, float *meansy, int k) {
    // initialize means to the mean of all data points and then add a small random number
    float tmpmeanx = 0;
    float tmpmeany = 0;
    for (int i = 0; i < values; i++) {
        tmpmeanx += pointsx[i];
        tmpmeany += pointsy[i];
    }
    for (int i = 0; i < k; i++) {
        meansx[i] = tmpmeanx / values + drand48();
        meansy[i] = tmpmeanx / values + drand48();
    }

    // these variables are accumulated for the update step
    // for each cluster we accumulate the x and y values of all points in the cluster
    // and we count the number of points per cluster
    float tmpx[k];      // temporary accumulator for x coordinates for each cluster 
    float tmpy[k];      // temporary accumulator for y coordinates for each cluster
    float sizes[k];     // number of data points in each cluster (i.e. the size of each cluster)

    // these variables are used as temporary values when assigning a point to it's closest cluster center
    float min_distance; // minimal distance of a point to it's closest cluster center
    int cluster;        // the closest cluster of a point
    float dist;         // distance of a point to a cluster

    // k-means Lloyd-Algorithm
    for (int e = 0; e < EPISODES; e++) {
        // reset accumulator values
        memset(tmpx, 0, sizeof(float) * k);
        memset(tmpy, 0, sizeof(float) * k);
        memset(sizes, 0, sizeof(float) * k);
        
        // STEP 1: assign all points to their closest cluster center
        // for all data points
        for (int i = 0; i < values; i++) {
            min_distance = INFINITY;
            cluster = -1;
            // find the closest cluster center
            for (int d = 0; d < k; d++) {
                if ((dist = distance(pointsx[i], pointsy[i], meansx[d], meansy[d])) 
                        < min_distance) {
                    min_distance = dist;
                    cluster = d;
                }
            }
            // add the point coordinates to the accumulator variables
            tmpx[cluster] += pointsx[i];
            tmpy[cluster] += pointsy[i];
            // and increase the number of points of the cluster
            sizes[cluster]++;
        }

        // STEP 2: update the centers by setting them to the means of their corresponding points
        for (int i = 0; i < k; i++) {
            if (sizes[i] > 0) {
                meansx[i] = tmpx[i] / sizes[i];
                meansy[i] = tmpy[i] / sizes[i];
            }
        }

    }
    return 0;
}

/**
 * @brief       K-means algorithm.
 *
 * @param[in]   pointsx     x coordinates of input points
 * @param[in]   pointsy     y coordinates of input points
 * @param[in]   values      number of input point
 * @param[out]  meansx      x coordinates of the resulting means
 * @param[out]  meansy      y coordinates of the resulting means
 * @param[in]   k           number of means to be computed
 *
 * @return      Returns 0 for success, -1 otherwise.
 */
int kmeans_simd(float *pointsx, float *pointsy, unsigned int values, 
                 float *meansx, float *meansy, int k) {
    // initialize means to the mean of all data points and then add a small random number
    // (You are allowed to change this, if you want to.)
    float tmpmeanx = 0;
    float tmpmeany = 0;
    for (int i = 0; i < values; i++) {
        tmpmeanx += pointsx[i];
        tmpmeany += pointsy[i];
    }
    for (int i = 0; i < k; i++) {
        meansx[i] = tmpmeanx / values + drand48();
        meansy[i] = tmpmeanx / values + drand48();
    }

    // Initialize SIMD vectors for temporary accumulator variables and sizes
    __m256 tmpx[k];      // temporary accumulator for x coordinates for each cluster
    __m256 tmpy[k];      // temporary accumulator for y coordinates for each cluster
    __m256 sizes[k];     // number of data points in each cluster (i.e., the size of each cluster)

    // Initialize SIMD vectors for meansx and meansy
    __m256 simd_meansx = _mm256_load_ps(meansx);
    __m256 simd_meansy = _mm256_load_ps(meansy);

    // K-means Lloyd-Algorithm
    for (int e = 0; e < EPISODES; e++) {
        // Reset accumulator values
        for (int i = 0; i < k; i++) {
            tmpx[i] = _mm256_setzero_ps();
            tmpy[i] = _mm256_setzero_ps();
            sizes[i] = _mm256_setzero_ps();
        }

        // STEP 1: assign all points to their closest cluster center
        // For all data points
        for (int i = 0; i < values; i += SIMD_FLOATS) {
            // Load SIMD vectors for point coordinates
            __m256 simd_pointsx = _mm256_load_ps(&pointsx[i]);
            __m256 simd_pointsy = _mm256_load_ps(&pointsy[i]);

            // Initialize SIMD vectors for min_distance and cluster
            __m256 min_distance = _mm256_set1_ps(INFINITY);
            __m256i cluster = _mm256_set1_epi32(-1);

            // Find the closest cluster center
            for (int d = 0; d < k; d++) {
                // Load SIMD vectors for cluster means
                __m256 simd_meansx_d = _mm256_set1_ps(meansx[d]);
                __m256 simd_meansy_d = _mm256_set1_ps(meansy[d]);

                // Compute the squared distance between point and cluster center
                __m256 simd_dx = _mm256_sub_ps(simd_pointsx, simd_meansx_d);
                __m256 simd_dy = _mm256_sub_ps(simd_pointsy, simd_meansy_d);
                __m256 simd_distance = _mm256_add_ps(_mm256_mul_ps(simd_dx, simd_dx), _mm256_mul_ps(simd_dy, simd_dy));

                // Update min_distance and cluster if the distance is smaller
                __m256 simd_mask = _mm256_cmp_ps(simd_distance, min_distance, _CMP_LT_OQ);
                min_distance = _mm256_blendv_ps(min_distance, simd_distance, simd_mask);
                cluster = _mm256_blendv_ps(cluster, _mm256_set1_ps(d), _mm256_castps_si256(simd_mask));
            }

            // Update the accumulator variables for the closest cluster
            for (int d = 0; d < k; d++) {
                // Find points that belong to the current cluster
                __m256i simd_mask = _mm256_cmpeq_ps(cluster, _mm256_set1_ps(d));

                // Update tmpx and tmpy with point coordinates
                tmpx[d] = _mm256_add_ps(tmpx[d], _mm256_and_ps(simd_pointsx, _mm256_castsi256_ps(simd_mask)));
                tmpy[d] = _mm256_add_ps(tmpy[d], _mm256_and_ps(simd_pointsy, _mm256_castsi256_ps(simd_mask)));

                // Increase sizes if the point belongs to the current cluster
                sizes[d] = _mm256_add_ps(sizes[d], _mm256_and_ps(_mm256_set1_ps(1), _mm256_castsi256_ps(simd_mask)));
            }
        }

        // STEP 2: Update means by computing the centroids of the clusters
        for (int d = 0; d < k; d++) {
            // Calculate new means by dividing the temporary accumulator variables by sizes
            __m256 simd_sizes = _mm256_div_ps(_mm256_set1_ps(values), sizes[d]);
            tmpx[d] = _mm256_div_ps(tmpx[d], sizes[d]);
            tmpy[d] = _mm256_div_ps(tmpy[d], sizes[d]);

            // Update meansx and meansy
            simd_meansx = _mm256_blendv_ps(simd_meansx, tmpx[d], _mm256_cmp_ps(sizes[d], _mm256_setzero_ps(), _CMP_NEQ_OQ));
            simd_meansy = _mm256_blendv_ps(simd_meansy, tmpy[d], _mm256_cmp_ps(sizes[d], _mm256_setzero_ps(), _CMP_NEQ_OQ));

            // Store the updated means
            _mm256_store_ps(&meansx[d], tmpx[d]);
            _mm256_store_ps(&meansy[d], tmpy[d]);
        }
    }

    // Store the final means
    _mm256_store_ps(meansx, simd_meansx);
    _mm256_store_ps(meansy, simd_meansy);

    return 0;
}


/**
 * @brief   Computes the difference between to timestamps.
 */
double timespec_diff(struct timespec ts1, struct timespec ts2) {
    return (double)(ts2.tv_sec - ts1.tv_sec) + (double)((ts2.tv_nsec - ts1.tv_nsec)/1e9f);
}

int main(int argc, const char *argv[])
{
    // seed random generator
    srand48(time(NULL));

    // variables for time measurements
    struct timespec ts0, ts1;
    double seq, simd;

    // read data points from CSV file
    float pointsx[VALUES] __attribute__((aligned(SIMD_BYTES)));
    float pointsy[VALUES] __attribute__((aligned(SIMD_BYTES)));
    readCSV(CLUSTER_FILE, pointsx, pointsy, VALUES);

    // two arrays to save the means
    float meansx[K] __attribute__((aligned(SIMD_BYTES)));
    float meansy[K] __attribute__((aligned(SIMD_BYTES)));

    // run k-means algorithm
    clock_gettime(CLOCK_REALTIME, &ts0);
    kmeans(pointsx, pointsy, VALUES, meansx, meansy, K); // sequential
    clock_gettime(CLOCK_REALTIME, &ts1);
    seq = timespec_diff(ts0, ts1);

    // run SIMD k-means algorithm
    clock_gettime(CLOCK_REALTIME, &ts0);
    kmeans_simd(pointsx, pointsy, VALUES, meansx, meansy, K); // simd
    clock_gettime(CLOCK_REALTIME, &ts1);
    simd = timespec_diff(ts0, ts1);

    printf("Timing: \n seq: %f \n simd: %f\n", seq, simd);

    // print means
    printf("Means (SIMD): \n");
    for (int i = 0; i < K; i++) {
        printf("%f %f\n", meansx[i], meansy[i]);
    }

    // save means to CSV file
    writeCSV(MEANS_FILE, meansx, meansy, K);
    return 0;
}
