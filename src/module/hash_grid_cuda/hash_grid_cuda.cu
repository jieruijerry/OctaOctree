#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define PRIME_X 1
#define PRIME_Y 19349663
#define PRIME_Z 83492791

#ifndef LEVELS
#define LEVELS 4
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 8
#endif

#ifndef LOG_HASHMAP_SIZE
#define LOG_HASHMAP_SIZE 19
#endif

#ifndef BASE_RESOLUTION
#define BASE_RESOLUTION 32
#endif

#ifndef PER_LEVEL_SCALE
#define PER_LEVEL_SCALE 2.0
#endif

#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif

#ifndef INTERP_PARALLEL
#define INTERP_PARALLEL  4
#endif

#ifndef INTERP_RATIO
#define INTERP_RATIO  0.33
#endif

#ifndef THREADS
#define THREADS 128
#endif


template <typename scalar_t>
struct scalar_t3 {
    scalar_t x, y, z;

    __host__ __device__ scalar_t3() : x(0), y(0), z(0) {}
    __host__ __device__ scalar_t3(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ scalar_t3 operator+(const scalar_t3& other) const {
        return scalar_t3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ scalar_t3 operator-(const scalar_t3& other) const {
        return scalar_t3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ scalar_t3 operator*(scalar_t s) const {
        return scalar_t3(x * s, y * s, z * s);
    }

    __host__ __device__ scalar_t3 operator/(scalar_t s) const {
        return scalar_t3(x / s, y / s, z / s);
    }

    __host__ __device__ scalar_t dot(const scalar_t3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ scalar_t norm() const {
        return sqrt(dot(*this));
    }

    __host__ __device__ scalar_t3 normalized() const {
        scalar_t n = norm();
        return n > 0 ? (*this) / n : scalar_t3(0, 0, 0);
    }

    __host__ __device__ int3 to_int3() const {
        return make_int3(
            static_cast<int>(floorf(x)),
            static_cast<int>(floorf(y)),
            static_cast<int>(floorf(z))
        );
    }
};


__device__ __forceinline__ int3 _corner_offset(int c) {
    return make_int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}


// Should be initialized in host function
__constant__ int resolutions[LEVELS];     // BASE_RESOLUTION * scale^level
__constant__ int grid_sizes[LEVELS];      // (res + 1)^3
__constant__ float grid_scales[LEVELS];   // INTERP_RATIO / res

__device__ __forceinline__ uint32_t _hash_index(int3 index, int level) {
    constexpr uint32_t MAX_HASH = 1 << LOG_HASHMAP_SIZE;

    int res1 = resolutions[level] + 1;
    int dense_max = grid_sizes[level];

    uint32_t hashed = 0;

    if (dense_max > MAX_HASH) {
        // Hash mode
        int64_t result =
            static_cast<int64_t>(index.x) * PRIME_X +
            static_cast<int64_t>(index.y) * PRIME_Y +
            static_cast<int64_t>(index.z) * PRIME_Z;
        hashed = static_cast<uint32_t>(llabs(result % MAX_HASH));
    } else {
        // Dense indexing mode
        int32_t dense_idx =
            res1 * res1 * index.x +
            res1 * index.y +
            index.z;
        hashed = static_cast<uint32_t>(labs(dense_idx % dense_max));
    }

    return hashed;
}


template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t *pos,
    const scalar_t **grids,
    scalar_t *result,
    int N
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int level = tid >> 3;   // divide by 8
    int corner = tid & 7;   // mod 8

    if (bid >= N || tid >= blockDim.x) return;

#if (LAYER_REDUCE == CONCAT)
    __shared__ scalar_t shared_out[LEVELS * DIMENSIONS];
    for (int i = tid; i < LEVELS * DIMENSIONS; i += blockDim.x) {
        shared_out[i] = 0.0f;
    }
#elif (LAYER_REDUCE == MEAN)
    __shared__ scalar_t shared_out[DIMENSIONS];
    for (int i = tid; i < DIMENSIONS; i += blockDim.x) {
        shared_out[i] = 0.0f;
    }
#endif
    __syncthreads();

    int res = resolutions[level];
    int grid_size = grid_sizes[level];
    
    const scalar_t *pos_ptr = pos + bid * 3;
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);

    // Get hash index
    scalar_t3<scalar_t> pos_grid = pos3 * res;
    int3 base = pos_grid.to_int3();
    scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
        pos_grid.x - base.x,
        pos_grid.y - base.y,
        pos_grid.z - base.z
    );
    int3 corner3 = _corner_offset(corner);
    int3 index3 = make_int3(
        base.x + corner3.x,
        base.y + corner3.y,
        base.z + corner3.z
    );
    uint32_t index = _hash_index(index3, level);
    assert(index < grid_size);

    // Calculate interpolation weight
    scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                      (corner3.y ? offset.y : (1 - offset.y)) *
                      (corner3.z ? offset.z : (1 - offset.z));

    const scalar_t *grid_ptr = grids[level];
    const scalar_t *grid = grid_ptr + index * DIMENSIONS;

#if (LAYER_REDUCE == CONCAT)
    for (int i = 0; i < DIMENSIONS; ++i) {
        atomicAdd(&shared_out[level * DIMENSIONS + i], grid[i] * weight);
    }
#elif (LAYER_REDUCE == MEAN)
    for (int i = 0; i < DIMENSIONS; ++i) {
        atomicAdd(&shared_out[i], grid[i] * weight / LEVELS);
    }
#endif
    __syncthreads();

    // Write to output
#if (LAYER_REDUCE == CONCAT)
    for (int i = tid; i < LEVELS * DIMENSIONS; i += blockDim.x) {
        result[bid * LEVELS * DIMENSIONS + i] = shared_out[i];
    }
#elif (LAYER_REDUCE == MEAN)
    for (int i = tid; i < DIMENSIONS; i += blockDim.x) {
        result[bid * DIMENSIONS + i] = shared_out[i];
    }
#endif
}


template <typename scalar_t>
__global__ void forward_layer_interp_kernel(
    const scalar_t *pos,
    const scalar_t *point_size,
    const scalar_t **grids,
    scalar_t *result,
    int N
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int ipid = tid >> 4;              // Interp parallel index = tid divide by 16
    int loffset = (tid >> 3) & 0x1;   // Lower or upper level
    int corner = tid & 7;             // Cornel index = tid mod 8

    if (bid >= N || tid >= blockDim.x) return;

    __shared__ scalar_t shared_out[INTERP_PARALLEL * 2 * DIMENSIONS];
    for (int i = tid; i < 2 * INTERP_PARALLEL * 2 * DIMENSIONS; i += blockDim.x) {
        shared_out[i] = 0.0f;
    }
    __syncthreads();

    const scalar_t *pos_ptr = pos + bid * INTERP_PARALLEL * 3 + ipid * 3;
    scalar_t psize_val = point_size[bid * INTERP_PARALLEL + ipid];
    scalar_t3<scalar_t> pos3 = scalar_t3<scalar_t>(pos_ptr[0], pos_ptr[1], pos_ptr[2]);

    // Get corresponding level and layer interpolation weight
    // Lower level means coarser, larger voxel size
    int level;
    scalar_t layer_weight = 0.0f;
    if (loffset) {  // Upper(finer) level, fit bottom-up
        level = LEVELS;
        for (int i = 0; i < LEVELS; ++i) {
            scalar_t voxel_size = grid_scales[i];
            if (psize_val > voxel_size) {
                level = i;
                break;
            }
        }
        scalar_t coarser_size = grid_scales[level - 1];
        scalar_t finer_size = grid_scales[level];
        layer_weight = (level == 0) ? 0.0f : ((level == LEVELS) ? 1.0f : (
            (coarser_size - psize_val) / (coarser_size - finer_size)));
    } else {        // Lower(coarser) level, fit top-down
        level = -1;
        for (int i = LEVELS - 1; i >= 0; --i) {
            scalar_t voxel_size = grid_scales[i];
            if (psize_val < voxel_size) {
                level = i;
                break;
            }
        }
        scalar_t coaser_size = grid_scales[level];
        scalar_t finer_size = grid_scales[level + 1];
        layer_weight = (level == LEVELS - 1) ? 0.0f : ((level == -1) ? 1.0f : (
            (psize_val - finer_size) / (coaser_size - finer_size)));
    }

    level = (level < 0) ? 0 : (level >= LEVELS ? LEVELS - 1 : level);

    // This thread has non-zero weight
    int res = resolutions[level];
    int grid_size = grid_sizes[level];

    // Get hash index
    scalar_t3<scalar_t> pos_grid = pos3 * res;
    int3 base = pos_grid.to_int3();
    scalar_t3<scalar_t> offset = scalar_t3<scalar_t>(
        pos_grid.x - base.x,
        pos_grid.y - base.y,
        pos_grid.z - base.z
    );
    int3 corner3 = _corner_offset(corner);
    int3 index3 = make_int3(
        base.x + corner3.x,
        base.y + corner3.y,
        base.z + corner3.z
    );
    uint32_t index = _hash_index(index3, level);
    assert(index < grid_size);

    // Calculate interpolation weight
    scalar_t weight = (corner3.x ? offset.x : (1 - offset.x)) *
                    (corner3.y ? offset.y : (1 - offset.y)) *
                    (corner3.z ? offset.z : (1 - offset.z));

    const scalar_t *grid_ptr = grids[level];
    const scalar_t *grid = grid_ptr + index * DIMENSIONS;

    for (int i = 0; i < DIMENSIONS; ++i) {
        atomicAdd(&shared_out[ipid * DIMENSIONS + i], grid[i] * weight * layer_weight);
    }
    __syncthreads();

    // Write to output
    for (int i = tid; i < INTERP_PARALLEL * DIMENSIONS; i += blockDim.x) {
        result[bid * INTERP_PARALLEL * DIMENSIONS + i] = shared_out[i];
    }
}


// Host functions

template<typename scalar_t>
const scalar_t **tensor_list_to_device_ptrs(const std::vector<torch::Tensor>& tensors) {
    size_t n = tensors.size();
    std::vector<const scalar_t *> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].contiguous().data_ptr<scalar_t>();
    }

    const scalar_t **device_ptrs;
    cudaMalloc(&device_ptrs, sizeof(const scalar_t *) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(const scalar_t *) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}

void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
) {
    auto N = pos.size(0);

    // Initalize resolution & grid size table
    int res_table[LEVELS];
    int grid_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);

    // Launch device function
    // int n_threads = THREADS;
    // int n_blocks = (N * L * 8 * D + THREADS - 1) / THREADS;
    int n_blocks = N;
    int n_threads = 8 * LEVELS;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "forward_kernel", ([&] {
        // Move grids vector/list to device pointer array
        const scalar_t **device_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);

        forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            device_ptrs,
            result.data_ptr<scalar_t>(),
            N
        );

        // Free device pointer array
        cudaFree(device_ptrs);
    }));
}

void launch_forward_layer_interp(
    const torch::Tensor pos,
    const torch::Tensor point_size,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
) {
    auto N = pos.size(0);

    // Initalize resolution & grid size table
    int res_table[LEVELS];
    int grid_table[LEVELS];
    float scale_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        res_table[i] = res;
        grid_table[i] = (res + 1) * (res + 1) * (res + 1);
        scale_table[i] = INTERP_RATIO / res;
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_scales, scale_table, sizeof(float) * LEVELS);

    // Launch device function
    int n_blocks = (N + INTERP_PARALLEL - 1) / INTERP_PARALLEL;
    int n_threads = INTERP_PARALLEL * 2 * 8;

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "forward_layer_interp_kernel", ([&] {
        // Move grids vector/list to device pointer array
        const scalar_t **device_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);

        forward_layer_interp_kernel<scalar_t><<<n_blocks, n_threads>>>(
            pos.data_ptr<scalar_t>(),
            point_size.data_ptr<scalar_t>(),
            device_ptrs,
            result.data_ptr<scalar_t>(),
            N
        );

        // Free device pointer array
        cudaFree(device_ptrs);
    }));
}