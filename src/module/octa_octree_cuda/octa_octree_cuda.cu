#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#ifndef LEVELS
#define LEVELS 8
#endif

#ifndef L_MIN
#define L_MIN 0
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 4
#endif

#ifndef DIMENSIONS_OUT
#define DIMENSIONS_OUT DIMENSIONS
#endif

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif

#ifndef THREADS
#define THREADS 256
#endif

__constant__ int directional_resolutions[LEVELS];
__constant__ int feature_depths[LEVELS];
__constant__ int64_t level_entries[LEVELS];
__constant__ int64_t feature_hashmap_size;
__constant__ float grid_scales[LEVELS * 3];

template <typename scalar_t>
struct scalar_t2 {
    scalar_t x, y;

    __host__ __device__ scalar_t2() : x(0), y(0) {}
    __host__ __device__ scalar_t2(scalar_t x_, scalar_t y_) : x(x_), y(y_) {}
};

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
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(scalar_t x) {
    return x < scalar_t(0) ? -x : x;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sign_val(scalar_t x) {
    return (x > scalar_t(0)) - (x < scalar_t(0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t floor_val(scalar_t x) {
    return floor(x);
}

__device__ __forceinline__ int3 corner_offset(int c) {
    return make_int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t3<scalar_t> normalize3(const scalar_t3<scalar_t>& v) {
    scalar_t n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z) + scalar_t(1e-8);
    return v / n;
}

template <typename scalar_t>
__device__ __forceinline__ bool is_invalid_disp(scalar_t x) {
    return isnan(static_cast<double>(x)) || isinf(static_cast<double>(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t2<scalar_t> octa_dir_to_uv(const scalar_t3<scalar_t>& dir_in) {
    scalar_t denom = abs_val(dir_in.x) + abs_val(dir_in.y) + abs_val(dir_in.z) + scalar_t(1e-8);
    scalar_t x = dir_in.x / denom;
    scalar_t y = dir_in.y / denom;
    scalar_t z = dir_in.z / denom;

    scalar_t u = (z > scalar_t(0)) ? (scalar_t(1) - abs_val(y)) * sign_val(x) : x;
    scalar_t v = (z > scalar_t(0)) ? (scalar_t(1) - abs_val(x)) * sign_val(y) : y;

    u = max(scalar_t(-1 + 1e-6), min(scalar_t(1 - 1e-6), u));
    v = max(scalar_t(-1 + 1e-6), min(scalar_t(1 - 1e-6), v));
    return scalar_t2<scalar_t>(u, v);
}

template <typename scalar_t>
__device__ __forceinline__ void octa_lerp(
    int base_u,
    int base_v,
    scalar_t frac_u,
    scalar_t frac_v,
    int resolution,
    int2 dir_idx[3],
    scalar_t dir_weight[3]
) {
    bool slash = ((base_u < 0) ^ (base_v < 0));
    bool ugtv = frac_u > frac_v;
    bool upvgt1 = (frac_u + frac_v) > scalar_t(1);

    int2 verts[3];

    if (slash) {
        if (ugtv) {
            verts[0] = make_int2(base_u, base_v);
            verts[1] = make_int2(base_u + 1, base_v + 1);
            verts[2] = make_int2(base_u + 1, base_v);

            dir_weight[0] = scalar_t(1) - frac_u;
            dir_weight[1] = frac_v;
            dir_weight[2] = frac_u - frac_v;
        } else {
            verts[0] = make_int2(base_u, base_v);
            verts[1] = make_int2(base_u + 1, base_v + 1);
            verts[2] = make_int2(base_u, base_v + 1);

            dir_weight[0] = scalar_t(1) - frac_v;
            dir_weight[1] = frac_u;
            dir_weight[2] = frac_v - frac_u;
        }
    } else {
        if (upvgt1) {
            verts[0] = make_int2(base_u + 1, base_v);
            verts[1] = make_int2(base_u, base_v + 1);
            verts[2] = make_int2(base_u + 1, base_v + 1);

            dir_weight[0] = scalar_t(1) - frac_v;
            dir_weight[1] = scalar_t(1) - frac_u;
            dir_weight[2] = frac_u + frac_v - scalar_t(1);
        } else {
            verts[0] = make_int2(base_u + 1, base_v);
            verts[1] = make_int2(base_u, base_v + 1);
            verts[2] = make_int2(base_u, base_v);

            dir_weight[0] = frac_u;
            dir_weight[1] = frac_v;
            dir_weight[2] = scalar_t(1) - frac_u - frac_v;
        }
    }

    for (int k = 0; k < 3; ++k) {
        int u = verts[k].x;
        int v = verts[k].y;

        bool left_right = abs(u) == (resolution / 2);
        bool top_bottom = abs(v) == (resolution / 2);
        bool slash_edge = ((u < 0) ^ (v < 0));
        bool corner = left_right && top_bottom;

        if (corner) {
            u = abs(u);
            v = abs(v);
        } else if (slash_edge && left_right) {
            v = -v;
        } else if (slash_edge && top_bottom) {
            u = -u;
        }

        u += resolution / 2;
        v += resolution / 2;
        dir_idx[k] = make_int2(u, v);
    }
}

__device__ __forceinline__ int64_t feature_flat_index(
    int64_t spatial_idx,
    int dir_u,
    int dir_v,
    int dir_grid_size,
    int level
) {
    if (level_entries[level] > feature_hashmap_size) {
        return (
            spatial_idx +
            19349663LL * static_cast<int64_t>(dir_u) +
            83492791LL * static_cast<int64_t>(dir_v)
        ) % feature_hashmap_size;
    }

    return spatial_idx * static_cast<int64_t>(dir_grid_size) +
           static_cast<int64_t>(dir_u * (directional_resolutions[level] + 1) + dir_v);
}

__device__ __forceinline__ int64_t dense_morton_key(int x, int y, int z, int depth_bits) {
    int64_t key = 0;
    for (int bit = 0; bit < depth_bits; ++bit) {
        key |= (static_cast<int64_t>((x >> bit) & 1) << (3 * bit + 2));
        key |= (static_cast<int64_t>((y >> bit) & 1) << (3 * bit + 1));
        key |= (static_cast<int64_t>((z >> bit) & 1) << (3 * bit + 0));
    }
    return key;
}

__device__ __forceinline__ int32_t search_nempty_children(
    const int32_t** children,
    int x,
    int y,
    int z,
    int depth
) {
    const int coarse_shift = depth - L_MIN;
    int32_t node_idx = children[0][dense_morton_key(
        x >> coarse_shift,
        y >> coarse_shift,
        z >> coarse_shift,
        L_MIN
    )];

    for (int abs_depth = L_MIN + 1; abs_depth <= depth && node_idx >= 0; ++abs_depth) {
        const int shift = depth - abs_depth;
        const int octant =
            (((x >> shift) & 1) << 2) |
            (((y >> shift) & 1) << 1) |
            (((z >> shift) & 1) << 0);
        node_idx = children[abs_depth - L_MIN][static_cast<int64_t>(node_idx) * 8 + octant];
    }

    return node_idx;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sample_disp_value(
    const scalar_t* disp,
    int bid,
    int level,
    int disp_stride
) {
    return disp_stride == 0 ? disp[bid] : disp[bid * disp_stride + level];
}

template <typename scalar_t>
__global__ void query_forward_kernel(
    const scalar_t* pos,
    const int32_t** children,
    const scalar_t* dir,
    const scalar_t* disp,
    const scalar_t** features,
    scalar_t* result,
    int N,
    int disp_stride
) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * LEVELS;

    if (thread_id >= total) {
        return;
    }
    const int bid = thread_id / LEVELS;
    const int level = thread_id % LEVELS;

    const scalar_t* dir_ptr = dir + bid * 3;
    const scalar_t* pos_ptr = pos + bid * 3;
    const scalar_t disp_val = sample_disp_value(disp, bid, level, disp_stride);
    scalar_t3<scalar_t> dir3(dir_ptr[0], dir_ptr[1], dir_ptr[2]);
    const int dir_res = directional_resolutions[level];
    const float* grid_scale = grid_scales + level * 3;
    const int dir_grid_size = (dir_res + 1) * (dir_res + 1);
    const int depth = feature_depths[level];
    const int spatial_bound = 1 << depth;
    const scalar_t spatial_scale = static_cast<scalar_t>(spatial_bound);

    const scalar_t x = pos_ptr[0] * spatial_scale - scalar_t(0.5);
    const scalar_t y = pos_ptr[1] * spatial_scale - scalar_t(0.5);
    const scalar_t z = pos_ptr[2] * spatial_scale - scalar_t(0.5);
    const int xi = static_cast<int>(floor_val(x));
    const int yi = static_cast<int>(floor_val(y));
    const int zi = static_cast<int>(floor_val(z));
    const scalar_t3<scalar_t> xyzf3(
        x - static_cast<scalar_t>(xi),
        y - static_cast<scalar_t>(yi),
        z - static_cast<scalar_t>(zi)
    );

    const scalar_t* feat = features[level];
    scalar_t out[DIMENSIONS_OUT];
    for (int d = 0; d < DIMENSIONS_OUT; ++d) {
        out[d] = scalar_t(0);
    }

    bool corner_valid[8];
    int32_t corner_idx[8];
    scalar_t corner_weights[8];
    scalar_t weight_sum = scalar_t(0);
    for (int corner = 0; corner < 8; ++corner) {
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight =
            (soff.x ? xyzf3.x : (scalar_t(1) - xyzf3.x)) *
            (soff.y ? xyzf3.y : (scalar_t(1) - xyzf3.y)) *
            (soff.z ? xyzf3.z : (scalar_t(1) - xyzf3.z));
        const int xn = xi + soff.x;
        const int yn = yi + soff.y;
        const int zn = zi + soff.z;
        const bool in_bound =
            xn >= 0 && yn >= 0 && zn >= 0 &&
            xn < spatial_bound && yn < spatial_bound && zn < spatial_bound;
        const int32_t spatial_idx = in_bound ? search_nempty_children(children, xn, yn, zn, depth) : -1;
        const bool valid = spatial_idx >= 0;

        corner_idx[corner] = spatial_idx;
        corner_valid[corner] = valid && corner_weight > scalar_t(0);
        corner_weights[corner] = corner_valid[corner] ? corner_weight : scalar_t(0);
        weight_sum += corner_weights[corner];
    }

    if (weight_sum <= scalar_t(0)) {
        return;
    }

    for (int corner = 0; corner < 8; ++corner) {
        if (!corner_valid[corner]) {
            continue;
        }

        const int64_t spatial_idx = corner_idx[corner];
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight = corner_weights[corner] / (weight_sum + scalar_t(1e-8));

        scalar_t3<scalar_t> perturb(
            (xyzf3.x - static_cast<scalar_t>(soff.x)) * static_cast<scalar_t>(grid_scale[0]),
            (xyzf3.y - static_cast<scalar_t>(soff.y)) * static_cast<scalar_t>(grid_scale[1]),
            (xyzf3.z - static_cast<scalar_t>(soff.z)) * static_cast<scalar_t>(grid_scale[2])
        );

        scalar_t3<scalar_t> sample_dir;
        if (is_invalid_disp(disp_val)) {
            sample_dir = dir3;
        } else {
            sample_dir = normalize3(dir3 + perturb * disp_val);
        }

        scalar_t2<scalar_t> uv = octa_dir_to_uv(sample_dir);
        scalar_t uv_scale = static_cast<scalar_t>(dir_res / 2);
        scalar_t u_scaled = uv.x * uv_scale;
        scalar_t v_scaled = uv.y * uv_scale;
        int uvi = static_cast<int>(floor_val(u_scaled));
        int vvi = static_cast<int>(floor_val(v_scaled));
        scalar_t uvf_u = u_scaled - static_cast<scalar_t>(uvi);
        scalar_t uvf_v = v_scaled - static_cast<scalar_t>(vvi);

        int2 dir_idx[3];
        scalar_t dir_weight[3];
        octa_lerp(uvi, vvi, uvf_u, uvf_v, dir_res, dir_idx, dir_weight);

        for (int k = 0; k < 3; ++k) {
            const int64_t flat_idx = feature_flat_index(
                spatial_idx,
                dir_idx[k].x,
                dir_idx[k].y,
                dir_grid_size,
                level
            );
            const scalar_t weight = corner_weight * dir_weight[k];
            const scalar_t* feat_ptr = feat + flat_idx * DIMENSIONS_OUT;
            for (int d = 0; d < DIMENSIONS_OUT; ++d) {
                out[d] += feat_ptr[d] * weight;
            }
        }
    }

    scalar_t* out_ptr = result + (bid * LEVELS + level) * DIMENSIONS_OUT;
    for (int d = 0; d < DIMENSIONS_OUT; ++d) {
        out_ptr[d] = out[d];
    }
}

template <typename scalar_t>
__global__ void query_backward_kernel(
    const scalar_t* grad_output,
    const scalar_t* pos,
    const int32_t** children,
    const scalar_t* dir,
    const scalar_t* disp,
    const scalar_t** features,
    scalar_t** grad_features,
    scalar_t* grad_disp,
    int N,
    int disp_stride
) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * LEVELS;

    if (thread_id >= total) {
        return;
    }
    const int bid = thread_id / LEVELS;
    const int level = thread_id % LEVELS;

    const scalar_t* grad_out = grad_output + (bid * LEVELS + level) * DIMENSIONS_OUT;
    const scalar_t* dir_ptr = dir + bid * 3;
    const scalar_t* pos_ptr = pos + bid * 3;
    const scalar_t disp_val = sample_disp_value(disp, bid, level, disp_stride);
    scalar_t3<scalar_t> dir3(dir_ptr[0], dir_ptr[1], dir_ptr[2]);
    const int dir_res = directional_resolutions[level];
    const float* grid_scale = grid_scales + level * 3;
    const int dir_grid_size = (dir_res + 1) * (dir_res + 1);
    const int depth = feature_depths[level];
    const int spatial_bound = 1 << depth;
    const scalar_t spatial_scale = static_cast<scalar_t>(spatial_bound);

    const scalar_t x = pos_ptr[0] * spatial_scale - scalar_t(0.5);
    const scalar_t y = pos_ptr[1] * spatial_scale - scalar_t(0.5);
    const scalar_t z = pos_ptr[2] * spatial_scale - scalar_t(0.5);
    const int xi = static_cast<int>(floor_val(x));
    const int yi = static_cast<int>(floor_val(y));
    const int zi = static_cast<int>(floor_val(z));
    const scalar_t3<scalar_t> xyzf3(
        x - static_cast<scalar_t>(xi),
        y - static_cast<scalar_t>(yi),
        z - static_cast<scalar_t>(zi)
    );

    const scalar_t* feat = features[level];
    scalar_t* grad_feat = grad_features[level];

    bool corner_valid[8];
    int32_t corner_idx[8];
    scalar_t corner_weights[8];
    scalar_t weight_sum = scalar_t(0);
    for (int corner = 0; corner < 8; ++corner) {
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight =
            (soff.x ? xyzf3.x : (scalar_t(1) - xyzf3.x)) *
            (soff.y ? xyzf3.y : (scalar_t(1) - xyzf3.y)) *
            (soff.z ? xyzf3.z : (scalar_t(1) - xyzf3.z));
        const int xn = xi + soff.x;
        const int yn = yi + soff.y;
        const int zn = zi + soff.z;
        const bool in_bound =
            xn >= 0 && yn >= 0 && zn >= 0 &&
            xn < spatial_bound && yn < spatial_bound && zn < spatial_bound;
        const int32_t spatial_idx = in_bound ? search_nempty_children(children, xn, yn, zn, depth) : -1;
        const bool valid = spatial_idx >= 0;

        corner_idx[corner] = spatial_idx;
        corner_valid[corner] = valid && corner_weight > scalar_t(0);
        corner_weights[corner] = corner_valid[corner] ? corner_weight : scalar_t(0);
        weight_sum += corner_weights[corner];
    }

    if (weight_sum <= scalar_t(0)) {
        return;
    }

    scalar_t grad_disp_local = scalar_t(0);

    for (int corner = 0; corner < 8; ++corner) {
        if (!corner_valid[corner]) {
            continue;
        }

        const int64_t spatial_idx = corner_idx[corner];
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight = corner_weights[corner] / (weight_sum + scalar_t(1e-8));

        scalar_t3<scalar_t> perturb(
            (xyzf3.x - static_cast<scalar_t>(soff.x)) * static_cast<scalar_t>(grid_scale[0]),
            (xyzf3.y - static_cast<scalar_t>(soff.y)) * static_cast<scalar_t>(grid_scale[1]),
            (xyzf3.z - static_cast<scalar_t>(soff.z)) * static_cast<scalar_t>(grid_scale[2])
        );

        scalar_t3<scalar_t> dir_pert = dir3 + perturb * disp_val;
        scalar_t3<scalar_t> sample_dir = is_invalid_disp(disp_val) ? dir3 : normalize3(dir_pert);

        const scalar_t l1_denom =
            abs_val(sample_dir.x) + abs_val(sample_dir.y) + abs_val(sample_dir.z) + scalar_t(1e-8);
        const scalar_t oct_x = sample_dir.x / l1_denom;
        const scalar_t oct_y = sample_dir.y / l1_denom;
        const scalar_t oct_z = sample_dir.z / l1_denom;
        const scalar_t raw_u = (oct_z > scalar_t(0)) ?
            (scalar_t(1) - abs_val(oct_y)) * sign_val(oct_x) : oct_x;
        const scalar_t raw_v = (oct_z > scalar_t(0)) ?
            (scalar_t(1) - abs_val(oct_x)) * sign_val(oct_y) : oct_y;
        const scalar_t uv_u = max(scalar_t(-1 + 1e-6), min(scalar_t(1 - 1e-6), raw_u));
        const scalar_t uv_v = max(scalar_t(-1 + 1e-6), min(scalar_t(1 - 1e-6), raw_v));

        scalar_t uv_scale = static_cast<scalar_t>(dir_res / 2);
        scalar_t u_scaled = uv_u * uv_scale;
        scalar_t v_scaled = uv_v * uv_scale;
        int uvi = static_cast<int>(floor_val(u_scaled));
        int vvi = static_cast<int>(floor_val(v_scaled));
        scalar_t uvf_u = u_scaled - static_cast<scalar_t>(uvi);
        scalar_t uvf_v = v_scaled - static_cast<scalar_t>(vvi);

        int2 dir_idx[3];
        scalar_t dir_weight[3];
        octa_lerp(uvi, vvi, uvf_u, uvf_v, dir_res, dir_idx, dir_weight);

        scalar_t grad_dir_weight[3];
        for (int k = 0; k < 3; ++k) {
            const int64_t flat_idx = feature_flat_index(
                spatial_idx,
                dir_idx[k].x,
                dir_idx[k].y,
                dir_grid_size,
                level
            );
            const scalar_t weight = corner_weight * dir_weight[k];
            const scalar_t* feat_ptr = feat + flat_idx * DIMENSIONS_OUT;
            scalar_t* grad_feat_ptr = grad_feat + flat_idx * DIMENSIONS_OUT;

            scalar_t dot_grad_feat = scalar_t(0);
            for (int d = 0; d < DIMENSIONS_OUT; ++d) {
                atomicAdd(&grad_feat_ptr[d], grad_out[d] * weight);
                dot_grad_feat += grad_out[d] * feat_ptr[d];
            }
            grad_dir_weight[k] = corner_weight * dot_grad_feat;
        }

        if (is_invalid_disp(disp_val)) {
            continue;
        }

        const bool slash = ((uvi < 0) ^ (vvi < 0));
        const bool ugtv = uvf_u > uvf_v;
        const bool upvgt1 = (uvf_u + uvf_v) > scalar_t(1);
        scalar_t grad_frac_u = scalar_t(0);
        scalar_t grad_frac_v = scalar_t(0);

        if (slash) {
            if (ugtv) {
                grad_frac_u = -grad_dir_weight[0] + grad_dir_weight[2];
                grad_frac_v = grad_dir_weight[1] - grad_dir_weight[2];
            } else {
                grad_frac_u = grad_dir_weight[1] - grad_dir_weight[2];
                grad_frac_v = -grad_dir_weight[0] + grad_dir_weight[2];
            }
        } else {
            if (upvgt1) {
                grad_frac_u = -grad_dir_weight[1] + grad_dir_weight[2];
                grad_frac_v = -grad_dir_weight[0] + grad_dir_weight[2];
            } else {
                grad_frac_u = grad_dir_weight[0] - grad_dir_weight[2];
                grad_frac_v = grad_dir_weight[1] - grad_dir_weight[2];
            }
        }

        scalar_t grad_raw_u = grad_frac_u * uv_scale;
        scalar_t grad_raw_v = grad_frac_v * uv_scale;
        if (raw_u <= scalar_t(-1 + 1e-6) || raw_u >= scalar_t(1 - 1e-6)) {
            grad_raw_u = scalar_t(0);
        }
        if (raw_v <= scalar_t(-1 + 1e-6) || raw_v >= scalar_t(1 - 1e-6)) {
            grad_raw_v = scalar_t(0);
        }

        scalar_t grad_oct_x = scalar_t(0);
        scalar_t grad_oct_y = scalar_t(0);
        scalar_t grad_oct_z = scalar_t(0);
        if (oct_z > scalar_t(0)) {
            const scalar_t sign_xy = sign_val(oct_x) * sign_val(oct_y);
            grad_oct_x += -grad_raw_v * sign_xy;
            grad_oct_y += -grad_raw_u * sign_xy;
        } else {
            grad_oct_x += grad_raw_u;
            grad_oct_y += grad_raw_v;
        }

        const scalar_t dot_grad_sample =
            grad_oct_x * sample_dir.x +
            grad_oct_y * sample_dir.y +
            grad_oct_z * sample_dir.z;
        const scalar_t l1_denom_sq = l1_denom * l1_denom;
        scalar_t3<scalar_t> grad_sample_dir(
            grad_oct_x / l1_denom - sign_val(sample_dir.x) * dot_grad_sample / l1_denom_sq,
            grad_oct_y / l1_denom - sign_val(sample_dir.y) * dot_grad_sample / l1_denom_sq,
            grad_oct_z / l1_denom - sign_val(sample_dir.z) * dot_grad_sample / l1_denom_sq
        );

        const scalar_t norm0 = sqrt(
            dir_pert.x * dir_pert.x +
            dir_pert.y * dir_pert.y +
            dir_pert.z * dir_pert.z
        );
        const scalar_t norm = norm0 + scalar_t(1e-8);
        const scalar_t dot_grad_dir_pert =
            grad_sample_dir.x * dir_pert.x +
            grad_sample_dir.y * dir_pert.y +
            grad_sample_dir.z * dir_pert.z;
        scalar_t3<scalar_t> grad_dir_pert(
            grad_sample_dir.x / norm,
            grad_sample_dir.y / norm,
            grad_sample_dir.z / norm
        );
        if (norm0 > scalar_t(0)) {
            const scalar_t inv_norm = scalar_t(1) / (norm0 * norm * norm);
            grad_dir_pert.x -= dir_pert.x * dot_grad_dir_pert * inv_norm;
            grad_dir_pert.y -= dir_pert.y * dot_grad_dir_pert * inv_norm;
            grad_dir_pert.z -= dir_pert.z * dot_grad_dir_pert * inv_norm;
        }

        grad_disp_local +=
            grad_dir_pert.x * perturb.x +
            grad_dir_pert.y * perturb.y +
            grad_dir_pert.z * perturb.z;
    }

    const int grad_disp_idx = disp_stride == 0 ? bid : bid * disp_stride + level;
    atomicAdd(&grad_disp[grad_disp_idx], grad_disp_local);
}

template <typename scalar_t>
__global__ void accumulate_forward_kernel(
    const scalar_t* feature,
    const scalar_t* level_weights,
    const scalar_t* pos,
    const int32_t** children,
    const scalar_t* dir,
    const scalar_t* disp,
    scalar_t** features,
    scalar_t** counts,
    int N
) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * LEVELS;

    if (thread_id >= total) {
        return;
    }
    const int bid = thread_id / LEVELS;
    const int level = thread_id % LEVELS;

    const scalar_t* dir_ptr = dir + bid * 3;
    const scalar_t* pos_ptr = pos + bid * 3;
    const scalar_t disp_val = disp[bid];
    const scalar_t level_weight = level_weights[bid * LEVELS + level];
    const scalar_t* sample_feat = feature + (bid * LEVELS + level) * DIMENSIONS;
    scalar_t3<scalar_t> dir3(dir_ptr[0], dir_ptr[1], dir_ptr[2]);
    const int dir_res = directional_resolutions[level];
    const float* grid_scale = grid_scales + level * 3;
    const int dir_grid_size = (dir_res + 1) * (dir_res + 1);
    const int depth = L_MIN + level;
    const int spatial_bound = 1 << depth;
    const scalar_t spatial_scale = static_cast<scalar_t>(spatial_bound);

    const scalar_t x = pos_ptr[0] * spatial_scale - scalar_t(0.5);
    const scalar_t y = pos_ptr[1] * spatial_scale - scalar_t(0.5);
    const scalar_t z = pos_ptr[2] * spatial_scale - scalar_t(0.5);
    const int xi = static_cast<int>(floor_val(x));
    const int yi = static_cast<int>(floor_val(y));
    const int zi = static_cast<int>(floor_val(z));
    const scalar_t3<scalar_t> xyzf3(
        x - static_cast<scalar_t>(xi),
        y - static_cast<scalar_t>(yi),
        z - static_cast<scalar_t>(zi)
    );

    scalar_t* feat_accum = features[level];
    scalar_t* count_accum = counts[level];

    bool corner_valid[8];
    int32_t corner_idx[8];
    scalar_t corner_weights[8];
    scalar_t weight_sum = scalar_t(0);
    for (int corner = 0; corner < 8; ++corner) {
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight =
            (soff.x ? xyzf3.x : (scalar_t(1) - xyzf3.x)) *
            (soff.y ? xyzf3.y : (scalar_t(1) - xyzf3.y)) *
            (soff.z ? xyzf3.z : (scalar_t(1) - xyzf3.z));
        const int xn = xi + soff.x;
        const int yn = yi + soff.y;
        const int zn = zi + soff.z;
        const bool in_bound =
            xn >= 0 && yn >= 0 && zn >= 0 &&
            xn < spatial_bound && yn < spatial_bound && zn < spatial_bound;
        const int32_t spatial_idx = in_bound ? search_nempty_children(children, xn, yn, zn, depth) : -1;
        const bool valid = spatial_idx >= 0;

        corner_idx[corner] = spatial_idx;
        corner_valid[corner] = valid && corner_weight > scalar_t(0);
        corner_weights[corner] = corner_valid[corner] ? corner_weight : scalar_t(0);
        weight_sum += corner_weights[corner];
    }

    if (weight_sum <= scalar_t(0)) {
        return;
    }

    for (int corner = 0; corner < 8; ++corner) {
        if (!corner_valid[corner]) {
            continue;
        }

        const int64_t spatial_idx = corner_idx[corner];
        const int3 soff = corner_offset(corner);
        const scalar_t corner_weight = corner_weights[corner] / (weight_sum + scalar_t(1e-8));

        scalar_t3<scalar_t> perturb(
            (xyzf3.x - static_cast<scalar_t>(soff.x)) * static_cast<scalar_t>(grid_scale[0]),
            (xyzf3.y - static_cast<scalar_t>(soff.y)) * static_cast<scalar_t>(grid_scale[1]),
            (xyzf3.z - static_cast<scalar_t>(soff.z)) * static_cast<scalar_t>(grid_scale[2])
        );

        scalar_t3<scalar_t> sample_dir;
        if (is_invalid_disp(disp_val)) {
            sample_dir = dir3;
        } else {
            sample_dir = normalize3(dir3 + perturb * disp_val);
        }

        scalar_t2<scalar_t> uv = octa_dir_to_uv(sample_dir);
        scalar_t uv_scale = static_cast<scalar_t>(dir_res / 2);
        scalar_t u_scaled = uv.x * uv_scale;
        scalar_t v_scaled = uv.y * uv_scale;
        int uvi = static_cast<int>(floor_val(u_scaled));
        int vvi = static_cast<int>(floor_val(v_scaled));
        scalar_t uvf_u = u_scaled - static_cast<scalar_t>(uvi);
        scalar_t uvf_v = v_scaled - static_cast<scalar_t>(vvi);

        int2 dir_idx[3];
        scalar_t dir_weight[3];
        octa_lerp(uvi, vvi, uvf_u, uvf_v, dir_res, dir_idx, dir_weight);

        for (int k = 0; k < 3; ++k) {
            const int64_t flat_idx = spatial_idx * dir_grid_size +
                dir_idx[k].x * (dir_res + 1) + dir_idx[k].y;
            const scalar_t weight = level_weight * corner_weight * dir_weight[k];
            scalar_t* feat_ptr = feat_accum + flat_idx * DIMENSIONS;
            for (int d = 0; d < DIMENSIONS; ++d) {
                atomicAdd(&feat_ptr[d], sample_feat[d] * weight);
            }
            atomicAdd(&count_accum[flat_idx], weight);
        }
    }
}

template <typename scalar_t>
const scalar_t** tensor_list_to_device_ptrs(const std::vector<torch::Tensor>& tensors) {
    const size_t n = tensors.size();
    std::vector<const scalar_t*> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].contiguous().data_ptr<scalar_t>();
    }

    const scalar_t** device_ptrs = nullptr;
    cudaMalloc(&device_ptrs, sizeof(const scalar_t*) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(const scalar_t*) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}

template <typename scalar_t>
scalar_t** tensor_list_to_device_mut_ptrs(std::vector<torch::Tensor>& tensors) {
    const size_t n = tensors.size();
    std::vector<scalar_t*> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].data_ptr<scalar_t>();
    }

    scalar_t** device_ptrs = nullptr;
    cudaMalloc(&device_ptrs, sizeof(scalar_t*) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(scalar_t*) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}

void launch_query_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size_value,
    int disp_stride,
    const std::vector<torch::Tensor>& features,
    torch::Tensor result
) {
    const auto N = dir.size(0);
    const int total = static_cast<int>(N) * LEVELS;

    cudaMemcpyToSymbol(
        directional_resolutions,
        resolutions_tensor.data_ptr<int>(),
        sizeof(int) * LEVELS,
        0,
        resolutions_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(
        feature_depths,
        depths_tensor.data_ptr<int>(),
        sizeof(int) * LEVELS,
        0,
        depths_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(
        level_entries,
        entries_tensor.data_ptr<int64_t>(),
        sizeof(int64_t) * LEVELS,
        0,
        entries_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(feature_hashmap_size, &hashmap_size_value, sizeof(int64_t));
    cudaMemcpyToSymbol(
        grid_scales,
        grid_scales_tensor.data_ptr<float>(),
        sizeof(float) * LEVELS * 3,
        0,
        grid_scales_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );

    const int n_threads = THREADS;
    const int n_blocks = (total + n_threads - 1) / n_threads;

    AT_DISPATCH_FLOATING_TYPES(dir.scalar_type(), "query_forward_kernel", ([&] {
        const int32_t** device_child_ptrs = tensor_list_to_device_ptrs<int32_t>(children);
        const scalar_t** device_feature_ptrs = tensor_list_to_device_ptrs<scalar_t>(features);

        query_forward_kernel<scalar_t><<<n_blocks, n_threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            pos.data_ptr<scalar_t>(),
            device_child_ptrs,
            dir.data_ptr<scalar_t>(),
            disp.data_ptr<scalar_t>(),
            device_feature_ptrs,
            result.data_ptr<scalar_t>(),
            static_cast<int>(N),
            disp_stride
        );

        cudaFree(device_child_ptrs);
        cudaFree(device_feature_ptrs);
    }));
}

std::vector<torch::Tensor> launch_query_backward(
    const torch::Tensor grad_output,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    const torch::Tensor depths_tensor,
    const torch::Tensor resolutions_tensor,
    const torch::Tensor entries_tensor,
    int64_t hashmap_size_value,
    int disp_stride,
    const std::vector<torch::Tensor>& features
) {
    const auto N = dir.size(0);
    const int total = static_cast<int>(N) * LEVELS;

    cudaMemcpyToSymbol(
        directional_resolutions,
        resolutions_tensor.data_ptr<int>(),
        sizeof(int) * LEVELS,
        0,
        resolutions_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(
        feature_depths,
        depths_tensor.data_ptr<int>(),
        sizeof(int) * LEVELS,
        0,
        depths_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(
        level_entries,
        entries_tensor.data_ptr<int64_t>(),
        sizeof(int64_t) * LEVELS,
        0,
        entries_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );
    cudaMemcpyToSymbol(feature_hashmap_size, &hashmap_size_value, sizeof(int64_t));
    cudaMemcpyToSymbol(
        grid_scales,
        grid_scales_tensor.data_ptr<float>(),
        sizeof(float) * LEVELS * 3,
        0,
        grid_scales_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );

    std::vector<torch::Tensor> grad_features;
    grad_features.reserve(features.size());
    for (const auto& feature : features) {
        grad_features.push_back(torch::zeros_like(feature));
    }
    torch::Tensor grad_disp = torch::zeros_like(disp);

    const int n_threads = THREADS;
    const int n_blocks = (total + n_threads - 1) / n_threads;

    AT_DISPATCH_FLOATING_TYPES(dir.scalar_type(), "query_backward_kernel", ([&] {
        const int32_t** device_child_ptrs = tensor_list_to_device_ptrs<int32_t>(children);
        const scalar_t** device_feature_ptrs = tensor_list_to_device_ptrs<scalar_t>(features);
        scalar_t** device_grad_feature_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(grad_features);

        query_backward_kernel<scalar_t><<<n_blocks, n_threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            grad_output.data_ptr<scalar_t>(),
            pos.data_ptr<scalar_t>(),
            device_child_ptrs,
            dir.data_ptr<scalar_t>(),
            disp.data_ptr<scalar_t>(),
            device_feature_ptrs,
            device_grad_feature_ptrs,
            grad_disp.data_ptr<scalar_t>(),
            static_cast<int>(N),
            disp_stride
        );

        cudaFree(device_child_ptrs);
        cudaFree(device_feature_ptrs);
        cudaFree(device_grad_feature_ptrs);
    }));

    grad_features.push_back(grad_disp);
    return grad_features;
}

void launch_accumulate_forward(
    const torch::Tensor feature,
    const torch::Tensor level_weights,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& children,
    const torch::Tensor dir,
    const torch::Tensor disp,
    const torch::Tensor grid_scales_tensor,
    std::vector<torch::Tensor>& features,
    std::vector<torch::Tensor>& counts
) {
    const auto N = feature.size(0);
    const int total = static_cast<int>(N) * LEVELS;
    int dir_table[LEVELS];

    for (int i = 0; i < LEVELS; ++i) {
        const auto dir_size = features[i].size(1);
        const int dir_res = static_cast<int>(std::round(std::sqrt(static_cast<double>(dir_size)))) - 1;

        dir_table[i] = dir_res;
    }

    cudaMemcpyToSymbol(directional_resolutions, dir_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(
        grid_scales,
        grid_scales_tensor.data_ptr<float>(),
        sizeof(float) * LEVELS * 3,
        0,
        grid_scales_tensor.is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice
    );

    const int n_threads = THREADS;
    const int n_blocks = (total + n_threads - 1) / n_threads;

    AT_DISPATCH_FLOATING_TYPES(feature.scalar_type(), "accumulate_forward_kernel", ([&] {
        const int32_t** device_child_ptrs = tensor_list_to_device_ptrs<int32_t>(children);
        scalar_t** device_feature_accum_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(features);
        scalar_t** device_count_accum_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(counts);

        accumulate_forward_kernel<scalar_t><<<n_blocks, n_threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            feature.data_ptr<scalar_t>(),
            level_weights.data_ptr<scalar_t>(),
            pos.data_ptr<scalar_t>(),
            device_child_ptrs,
            dir.data_ptr<scalar_t>(),
            disp.data_ptr<scalar_t>(),
            device_feature_accum_ptrs,
            device_count_accum_ptrs,
            static_cast<int>(N)
        );

        cudaFree(device_child_ptrs);
        cudaFree(device_feature_accum_ptrs);
        cudaFree(device_count_accum_ptrs);
    }));
}
