import torch
import numpy as np

from typing import Union
import time

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def get_model_bbox(scene: mi.Scene, device: torch.device = "cuda") -> torch.Tensor:
    """
    Get the bounding box of the scene
    """
    bbox = scene.bbox()
    bbox = torch.tensor([bbox.min - 1e-2, bbox.max + 1e-2], dtype=torch.float32, device=device)
    return bbox

def normalize_pos(pos: torch.Tensor, bbox: torch.Tensor, isotropic=False) -> torch.Tensor:
    """
    Normalize positions to [0, 1] range.
    """
    if isotropic:
        return (pos - bbox[0]) / (bbox[1] - bbox[0]).max()
    return (pos - bbox[0]) / (bbox[1] - bbox[0])

def compute_areas(scene: mi.Scene):
    """
    Compute the areas of each mesh of the scene
    """
    shapes = scene.shapes_dr()
    areas = []
    for shape in shapes:
        # print("Mesh", shape.id(), ": is_emitter:", shape.is_emitter(), ", is_mesh:", shape.is_mesh(),
        #       ", smooth:", mi.has_flag(shape.bsdf().flags(), mi.BSDFFlags.Smooth))
        areas.append(shape.surface_area())
    areas = np.array(areas)[:, 0]
    areas = areas / areas.sum()
    return areas

def sample_si(scene: mi.Scene, num_samples: int, seed : int = 0) -> mi.SurfaceInteraction3f:
    # Set sampler
    areas = compute_areas(scene)
    l_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    l_sampler.seed(seed, num_samples)

    # Sample shape
    shape_sampler = mi.DiscreteDistribution(areas)
    shape_idx = shape_sampler.sample(l_sampler.next_1d(), True)
    shapes: mi.ShapePtr = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx, True)

    # Sample position
    ps = shapes.sample_position(0, l_sampler.next_2d(), True)
    si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si.shape = shapes
    si.t = mi.Float(0)
    si.prim_index = ps.pidx
    si.buv = ps.buv

    # Sample outgoing direction
    active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
    si.wi = dr.select(
        active_two_sided,
        mi.warp.square_to_uniform_sphere(l_sampler.next_2d()),
        mi.warp.square_to_cosine_hemisphere(l_sampler.next_2d()),
    )

    return si

def sample_si_pose(pose: dict, scene: mi.Scene, num_samples: int, seed : int = 0, rr_prob: float = 0.7) -> mi.SurfaceInteraction3f:
    # Set sampler
    sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, num_samples)

    # Generate random camera rays
    to_world = mi.Transform4f(pose["extrinsics"][None, ...])
    x_fov = pose["intrinsics"]["x_fov"]
    width = pose["intrinsics"]["width"]
    height = pose["intrinsics"]["height"]
    focal_length = 0.5 * width / np.tan(np.radians(x_fov) * 0.5)

    sample = sampler.next_2d() - 0.5
    d = dr.normalize(mi.Vector3f(-width * sample[0], height * sample[1], focal_length))
    
    ray = mi.Ray3f()
    ray.o = to_world.translation()
    ray.d = to_world @ d
    
    # Intersect rays with the scene
    active = mi.Bool(True)
    si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
    final_si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    bsdf_ctx = mi.BSDFContext()

    loop = mi.Loop(
        "sample LHS pose",
        lambda: (
            sampler,
            ray,
            si,
            final_si,
            active
        )
    )

    max_depth = 16
    loop.set_max_iterations(max_depth)

    while loop(active):
        si = scene.ray_intersect(ray, active)
        bsdf: mi.BSDF = si.bsdf()

        null_face = ~si.is_valid() | (
            (si.wi.z < 0) & ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        )

        # Sample ray
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
        )
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # Reset ray for invalid intersections
        sample = (sampler.next_2d() - 0.5)[null_face]
        d = dr.normalize(mi.Vector3f(-width * sample[0], height * sample[1], focal_length))
        ray.o[null_face] = to_world.translation()
        ray.d[null_face] = to_world @ d

        recorded = sampler.next_1d() > rr_prob & ~null_face

        final_si[active & recorded] = si
        active &= ~recorded
    
    final_si.wi = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())

    dr.eval(final_si)

    return final_si

def first_smooth(
    scene: mi.Scene,
    sampler: mi.Sampler,
    si_or_ray: Union[mi.SurfaceInteraction3f, mi.Ray3f],
    active: bool = True
) -> tuple[mi.SurfaceInteraction3f, mi.Color3f, bool]:

    with dr.suspend_grad():

        final_si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        active = mi.Bool(active)
        throughput = mi.Color3f(1.0)
        emission = mi.Color3f(0.0)
        depth = mi.UInt32(0)

        si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        if isinstance(si_or_ray, mi.Ray3f):
            si = scene.ray_intersect(si_or_ray, active)
        elif isinstance(si_or_ray, mi.SurfaceInteraction3f):
            si = si_or_ray

        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "first smooth surface",
            lambda: (
                sampler,
                si,
                final_si,
                active,
                throughput,
                emission,
                depth,
            )
        )

        max_depth = 16
        loop.set_max_iterations(max_depth)

        while loop(active):

            bsdf: mi.BSDF = si.bsdf()
            final_si[active] = si

            # Mask of rays to continue tracing
            spec_only = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Delta) & \
                       ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            
            mask_out = bsdf.eval_null_transmission(si)
            mask_out = (mask_out.x > 0) | (mask_out.y > 0) | (mask_out.z > 0)

            active &= si.is_valid() & (spec_only | mask_out) & (depth < max_depth)

            # Mask of rays that hit a null face
            null_face = ~si.is_valid() | (
                (si.wi.z < 0) & ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
            )

            # Rays that hit emissive surfaces
            current_emission = si.emitter(scene).eval(si)
            emissive = (current_emission.x > 0) | (current_emission.y > 0) | (current_emission.z > 0)

            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            emission += current_emission * throughput
            throughput[active] *= bsdf_weight
            throughput[null_face | emissive] = 0
            depth[si.is_valid()] += 1

            si = scene.ray_intersect(ray, active)

    # return final_si, throughput, null_face, spec_mask
    return final_si, throughput, emission, False


def extract_input(si: mi.SurfaceInteraction3f, flip_dir: bool = True, device: str = "cuda", dtype=torch.float32):
    """
    Extract input features from the surface interaction
    """
    dr.eval(si)

    wi = si.to_world(si.wi)
    n = si.sh_frame.n
    wr = 2 * dr.dot(wi, n) * n - wi
    si_view = mi.SurfaceInteraction3f(si)
    si_view.wi = mi.Point3f([0.353553, 0.353553, 0.866025])

    p = si.p.torch().to(device=device, dtype=dtype)
    n = n.torch().to(device=device, dtype=dtype)
    dir = (wr if flip_dir else wi).torch().to(device=device, dtype=dtype)
    albedo = si_view.bsdf().eval_diffuse_reflectance(si_view).torch().to(device=device, dtype=dtype)
    roughness = si.bsdf().eval_roughness(si).torch().unsqueeze(1).to(device=device, dtype=dtype)
    # active_side = (si.wi[2].torch() < 0).unsqueeze(1).to(device=device, dtype=dtype)
    active = si.is_valid().torch().bool().unsqueeze(1).to(device=device)
    
    p[n.isnan()] = 0.0
    n[n.isnan()] = 0.3
    dir[dir.isnan()] = 0.3

    return p, n, dir, albedo, roughness, active


def get_wr_itsc(si: mi.SurfaceInteraction3f, scene: mi.Scene) -> mi.SurfaceInteraction3f:
    # Trace secondary ray in the reflection direction, get intersection
    wi = si.to_world(si.wi)
    n = si.sh_frame.n
    wr = 2 * dr.dot(wi, n) * n - wi

    ray = si.spawn_ray(wr)
    si_wr = scene.ray_intersect(ray)

    return si_wr


def safe_wr_disp(scene: mi.Scene, si: mi.SurfaceInteraction3f, dtype=torch.float32) -> torch.Tensor:
    min_t = float(1e-3)
    si_wr = get_wr_itsc(si, scene)
    t = si_wr.t.torch().to(dtype=dtype)
    t = torch.nan_to_num(t, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)

    disp = torch.zeros_like(t)
    valid = t > 0
    disp[valid] = torch.reciprocal(t[valid].clamp(min=min_t))

    return disp.unsqueeze(-1)


def get_mc_itsc(
    si: mi.SurfaceInteraction3f,
    scene: mi.Scene,
    mask: torch.Tensor,
    rhs_num: int,
    seed: int = np.random.randint(0, 1000000)
):
    """
    Monte Carlo sampling RHS intersection over bsdf
    """
    # Gather glossy interactions
    indices = torch.nonzero(mask).squeeze().to(dtype=torch.int32)
    point_num = indices.shape[0]

    indices = dr.repeat(mi.Int(indices), rhs_num)
    si_rhs: mi.SurfaceInteraction3f = dr.gather(mi.SurfaceInteraction3f, si, indices)

    # Sample RHS interactions for glossy interactions
    r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    r_sampler.seed(seed, point_num * rhs_num)
    ctx = mi.BSDFContext()
    bsdf_sample, bsdf_weight = si_rhs.bsdf().sample(
        ctx, si_rhs,
        r_sampler.next_1d() * 0,  # Force sampling the specular lobe
        r_sampler.next_2d(),
        active=True,
    )

    ray = si_rhs.spawn_ray(si_rhs.to_world(bsdf_sample.wo))
    si_bsdf_first = scene.ray_intersect(ray)

    return si_bsdf_first, bsdf_sample, bsdf_weight


def flags_to_lobe(flags: torch.Tensor) -> torch.Tensor:
    """
    Convert BSDF flags to lobe index
    """
    diffuse_lobe = (flags & (0x006)) != 0
    glossy_lobe = (flags & (0x018)) != 0
    delta_lobe = (flags & (0x1E0)) != 0
    lobe = torch.cat([diffuse_lobe, glossy_lobe, delta_lobe], dim=-1)
    return lobe

def lobe_to_flags(lobe: torch.Tensor) -> torch.Tensor:
    """
    Convert lobe index to BSDF flags
    """
    diffuse_lobe = lobe[:, 0:1]
    glossy_lobe = lobe[:, 1:2]
    delta_lobe = lobe[:, 2:3]
    flags = (diffuse_lobe * 0x006) + (glossy_lobe * 0x018) + (delta_lobe * 0x1E0)
    return flags.to(torch.int32)


def barycentric_interp(
    point_positions: torch.Tensor,   # [batch_idx, xyz]
    vertex_positions: torch.Tensor,  # [batch_idx, vertex_idx, xyz]
) -> torch.Tensor:                   # [batch_idx, vertex_idx]
    v0 = vertex_positions[:, 0, :]
    v1 = vertex_positions[:, 1, :]
    v2 = vertex_positions[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    d = point_positions - v0
    det = torch.cross(e1, e2, dim=1).norm(dim=1)
    u = torch.cross(d, e2, dim=1).norm(dim=1) / det
    v = torch.cross(e1, d, dim=1).norm(dim=1) / det
    w = 1 - u - v
    
    return torch.stack([w, u, v], dim=1)


def interp_mesh_to_si(
    vertex_indices: torch.Tensor,     # [N, 3: vertex_idx]
    vertex_weights: torch.Tensor,     # [N, 3: vertex_idx]
    mesh_feature: torch.Tensor,       # [2, M, D]
    active_side: torch.Tensor,        # [N, 1]
) -> torch.Tensor:
    """
    Interpolate mesh feature onto vertices.
    """
    # Flatten mesh feature
    D = mesh_feature.shape[-1]
    
    # Interpolate mesh feature into si feature
    if active_side == None:
        # [N, 3: vertex_idx, D]
        si_feature = mesh_feature[:, vertex_indices.reshape(-1)].reshape(-1, 3, D)
    else:
        si_feature = mesh_feature[:, vertex_indices.reshape(-1)].reshape(2, -1, 3, D)
        # [N, 3: vertex_idx, D]
        si_feature = torch.where(active_side[..., None], si_feature[1], si_feature[0])
    
    # [N, D]
    si_feature = (si_feature * vertex_weights[..., None]).sum(dim=1)
    
    return si_feature