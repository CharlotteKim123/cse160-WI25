__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A^T B 

    int row = get_global_id(0);
    int col = get_global_id(1);

    if(row < numCRows && col < numACColumns){
        int sum = 0;
        for (int i = 0; i < numAColumns; i++){
                sum += A[i * numARows + row] * B[i * numBColumns + col];
        }
        C[row * numCColumns + col] = sum;
    }
}
