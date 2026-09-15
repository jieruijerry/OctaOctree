import torch
import numpy as np

from typing import Union
import time

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def render_pt(scene: mi.Scene, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, incident_radiance=False):

    si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    ray: mi.Ray3f = mi.Ray3f(si.p + si.n * 1e-4, si.to_world(si.wi))
    active = mi.Bool(True)
    throughput = mi.Color3f(1.0)
    result = mi.Color3f(0.0)
    depth = mi.UInt32(0)

    valid_ray = mi.Bool(scene.environment() is not None)

    # Variables caching information from the previous bounce
    prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Bool(True)
    bsdf_ctx = mi.BSDFContext()

    loop = mi.Loop(
        "Path Tracer",
        lambda: (
            sampler,
            ray,
            si,
            throughput,
            result,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
        )
    )

    max_depth = 8
    loop.set_max_iterations(max_depth)

    while loop(active):

        if incident_radiance:
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        # ---------------------- Direct emission ----------------------

        mis_weight = mi.Float(1.0)

        mis_weight[~prev_bsdf_delta] = 0.0

        ds: mi.DirectionSample3f = mi.DirectionSample3f(
            scene, si, prev_si)
        em_pdf = mi.Float(0.0)

        em_pdf = scene.pdf_emitter_direction(
            prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = dr.detach(
            dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
        mis_weight[~prev_bsdf_delta] = mis_bsdf

        result = dr.fma(
            throughput,
            si.emitter(scene).eval(si) * mis_weight,
            result
        )

        active_next = ((depth + 1) < max_depth) & si.is_valid()

        bsdf: mi.BSDF = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )

        wo = si.to_local(ds.d)

        # ------ Evaluate BSDF * cos(theta) and sample direction -------

        sample1 = sampler.next_1d()
        sample2 = sampler.next_2d()

        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
            bsdf_ctx, si, wo, sample1, sample2
        )

        # --------------- Emitter sampling contribution ----------------

        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        mi_em = mi.Float(1.0)
        mi_em = dr.select(ds.delta, 1.0, dr.detach(
            dr.select(ds.pdf > 0, ds.pdf / (ds.pdf + bsdf_pdf), 0)))

        result[active_em] = dr.fma(
            throughput, bsdf_val * em_weight * mi_em, result)

        # ---------------------- BSDF sampling ----------------------

        bsdf_weight = si.to_world_mueller(
            bsdf_weight, -bsdf_sample.wo, si.wi)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        valid_ray |= (
            active
            & si.is_valid()
            & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
        )

        prev_si = mi.Interaction3f(si)
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1

        throughput_max = dr.max(throughput)

        rr_prop = dr.minimum(throughput_max, 0.95)
        rr_active = (depth >= 3)
        rr_continue = (sampler.next_1d() < rr_prop)

        throughput[rr_active] *= dr.rcp(rr_prop)

        active = (
            active_next & (~rr_active | rr_continue) & (
                dr.neq(throughput_max, 0.0))
        )

        if not incident_radiance:
            si = scene.ray_intersect(ray, active)

    return dr.select(valid_ray, result, 0.0), valid_ray

def render_pt_spp(scene: mi.Scene, si: mi.SurfaceInteraction3f, spp: int = 1, incident_radiance=False):
    # Calculate number of passes
    size = dr.width(si.p)
    spp_per_pass = spp
    n_passes = 1
    wavefront_size = size * spp
    wavefront_limit = 2 ** 28

    if wavefront_size > wavefront_limit:
        n_passes = (wavefront_size + wavefront_limit - 1) // wavefront_limit
        spp_per_pass = spp // n_passes

    # Render
    indices = dr.repeat(dr.arange(mi.Int, 0, size), spp_per_pass)
    si_multi = dr.gather(mi.SurfaceInteraction3f, si, indices)
    
    # Accumulate passes
    sampler: mi.Sampler = mi.load_dict({"type": "independent"})
    result = dr.zeros(mi.Color3f, size * spp_per_pass)
    for _ in range(n_passes):
        sampler.seed(np.random.randint(0, 10000000), dr.width(si_multi.p))
        result += render_pt(scene, sampler, si_multi, incident_radiance=incident_radiance)[0]
    result = result / n_passes

    # Accumulate spp per pass
    result = dr.block_sum(result, spp_per_pass) / spp_per_pass

    return result


def render_pt_with_lobe(scene: mi.Scene, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, lobe_var: mi.Float, incident_radiance=False):

    si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    ray: mi.Ray3f = mi.Ray3f(si.p + si.n * 1e-4, si.to_world(si.wi))
    active = mi.Bool(True)
    throughput = mi.Color3f(1.0)
    result = mi.Color3f(0.0)
    depth = mi.UInt32(0)

    valid_ray = mi.Bool(scene.environment() is not None)

    # Variables caching information from the previous bounce
    prev_si: mi.Interaction3f = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Bool(True)
    bsdf_ctx = mi.BSDFContext()

    loop = mi.Loop(
        "Path Tracer",
        lambda: (
            sampler,
            ray,
            si,
            throughput,
            result,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
        )
    )

    max_depth = 8
    loop.set_max_iterations(max_depth)

    while loop(active):

        if incident_radiance:
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        # ---------------------- Direct emission ----------------------

        mis_weight = mi.Float(1.0)

        mis_weight[~prev_bsdf_delta] = 0.0

        ds: mi.DirectionSample3f = mi.DirectionSample3f(
            scene, si, prev_si)
        em_pdf = mi.Float(0.0)

        em_pdf = scene.pdf_emitter_direction(
            prev_si, ds, ~prev_bsdf_delta)

        mis_bsdf = dr.detach(
            dr.select(prev_bsdf_pdf > 0, prev_bsdf_pdf / (prev_bsdf_pdf + em_pdf), 0))
        mis_weight[~prev_bsdf_delta] = mis_bsdf

        result = dr.fma(
            throughput,
            si.emitter(scene).eval(si) * mis_weight,
            result
        )

        active_next = ((depth + 1) < max_depth) & si.is_valid()

        bsdf: mi.BSDF = si.bsdf(ray)

        # ---------------------- Emitter sampling ----------------------

        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )

        wo = si.to_local(ds.d)

        # ------ Evaluate BSDF * cos(theta) and sample direction -------

        sample1 = dr.select(depth < 1, lobe_var, sampler.next_1d())
        sample2 = sampler.next_2d()

        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
            bsdf_ctx, si, wo, sample1, sample2
        )

        # --------------- Emitter sampling contribution ----------------

        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

        mi_em = mi.Float(1.0)
        mi_em = dr.select(ds.delta, 1.0, dr.detach(
            dr.select(ds.pdf > 0, ds.pdf / (ds.pdf + bsdf_pdf), 0)))

        result[active_em] = dr.fma(
            throughput, bsdf_val * em_weight * mi_em, result)

        # ---------------------- BSDF sampling ----------------------

        bsdf_weight = si.to_world_mueller(
            bsdf_weight, -bsdf_sample.wo, si.wi)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # ------ Update loop variables based on current interaction ------

        throughput *= bsdf_weight
        valid_ray |= (
            active
            & si.is_valid()
            & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
        )

        prev_si = mi.Interaction3f(si)
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(
            bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # -------------------- Stopping criterion ---------------------

        depth[si.is_valid()] += 1

        throughput_max = dr.max(throughput)

        rr_prop = dr.minimum(throughput_max, 0.95)
        rr_active = (depth >= 3)
        rr_continue = (sampler.next_1d() < rr_prop)

        throughput[rr_active] *= dr.rcp(rr_prop)

        active = (
            active_next & (~rr_active | rr_continue) & (
                dr.neq(throughput_max, 0.0))
        )

        if not incident_radiance:
            si = scene.ray_intersect(ray, active)

    return dr.select(valid_ray, result, 0.0), valid_ray