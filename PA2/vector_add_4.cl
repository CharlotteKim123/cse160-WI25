__kernel void vectorAdd(__global const int *a, __global const int *b,
			            __global const int *c, __global const int *d,
                        __global int *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here
    int i = get_global_id(0);
    if(i >= size){
    	return 0;
    }
    result[i] = a[i] + b[i] c[i] + d[i];
}
