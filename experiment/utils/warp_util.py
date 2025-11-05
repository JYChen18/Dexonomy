import warp as wp


@wp.kernel
def add_arr1D_wp(a: wp.array(), 
                 b: wp.array(),
                 c: wp.array()):
  tid = wp.tid()
  c[tid] = a[tid] + b[tid]
  
@wp.kernel
def axbyz_arr1D_wp(x: wp.array(),
                  a: wp.float32,
                  y: wp.array(),
                  b: wp.float32,
                  z: wp.array()):
  tid = wp.tid()
  z[tid] = a * x[tid] + b * y[tid]

@wp.kernel
def axpy_arr1D_wp(x: wp.array(),
                  a: wp.float32,
                  y: wp.array()):
  tid = wp.tid()
  y[tid] += a * x[tid]