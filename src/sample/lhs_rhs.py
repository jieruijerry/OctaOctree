import torch
import numpy as np

from typing import Union
import time

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

from src.sample.util import *


class LHSRHS:
    def __init__(self, scene: mi.Scene, point_num: int, dirs_per_point: int):
        # Scene information
        self.scene: mi.Scene = scene
        self.point_num: int = point_num
        self.dirs_per_point: int = dirs_per_point
        # LHS (Left Hand Side) interaction
        self.si_lhs: mi.SurfaceInteraction3f = None
        # Duplicated LHS interaction for each direction
        self.si_rhs: mi.SurfaceInteraction3f = None
        # RHS interaction according to BSDF sampling
        self.si_bsdf: mi.SurfaceInteraction3f = None
        # RHS interaction according to emitter or sub-emitter sampling
        self.si_emit: mi.SurfaceInteraction3f = None
        # Main reflection interaction
        self.si_wr: mi.SurfaceInteraction3f = None

        ## Light transport weights
        # BSDF weight    (f(d_b) / p_b(d_b)) (L_i(d_b) to be multiplied)
        self._bsdf_weight = None
        # Emitter weight (L_e(d_e) / p_e(d_e)) (f(d_e) to be multiplied)
        self._emit_weight = None
        self._f_d_e = None

        ## Direction samples
        # Outgoing direction from BSDF sampling
        self._bsdf_sample: mi.BSDFSample3f = None
        # Outgoing direction from emitter sampling
        self._emit_sample: mi.DirectionSample3f = None

        ## PDF values for MIS
        # PDF of BSDF sampling given direction from BSDF sampling
        self._p_b_d_b = None
        # PDF of BSDF sampling given direction from emitter sampling
        self._p_b_d_e = None
        # PDF of emitter sampling given direction from BSDF sampling
        self._p_e_d_b = None
        # PDF of emitter sampling given direction from emitter sampling
        self._p_e_d_e = None
        
        ## MIS weights
        # MIS weight for BSDF sampling
        self._mis_bsdf = None
        # MIS weight for emitter sampling
        self._mis_emit = None

    def sample(self,
        seed : int = 0,
        si_lhs : mi.SurfaceInteraction3f = None,
        pose : dict = None
    ):
        # Sampling process
        if si_lhs is not None:
            self.si_lhs = si_lhs
            self.si_rhs = self.get_rhs_from_lhs(si_lhs)
        else:
            self.sample_lhs(seed, pose=pose)
        # self.get_wr_itsc()
        self.sample_bsdf_rhs(seed)
        self.sample_emit_rhs(seed)
        self.eval_pdf()

    def render(self,
        bsdf_color: torch.Tensor,
        sub_emit_color: torch.Tensor = None
    ):
        t0 = time.time()
        # LHS emitter mask
        emitter_mask = self._emission > 1e-6
        t1 = time.time()

        # Render LHS color given BSDF & emitter sampling RHS
        bsdf_thp = self._bsdf_thp.reshape(-1, self.dirs_per_point, 3)
        bsdf_em = self._bsdf_em.reshape(-1, self.dirs_per_point, 3)
        bsdf_color = bsdf_color.reshape(-1, self.dirs_per_point, 3)
        bsdf_color = torch.nan_to_num(
            bsdf_color * bsdf_thp + bsdf_em,
            nan=0.0, posinf=0.0, neginf=0.0
        )
        t2 = time.time()

        bsdf_weight = self._bsdf_weight.reshape(-1, self.dirs_per_point, 3)
        mis_bsdf_weight = self._mis_bsdf.reshape(-1, self.dirs_per_point, 1)
        L_o_d_b = torch.nan_to_num(bsdf_color * bsdf_weight * mis_bsdf_weight, nan=0.0, posinf=0.0, neginf=0.0)
        t3 = time.time()

        f_d_e = self._f_d_e.reshape(-1, self.dirs_per_point, 3)
        emit_weight = self._emit_weight.reshape(-1, self.dirs_per_point, 3)
        mis_emit_weight = self._mis_emit.reshape(-1, self.dirs_per_point, 1)
        L_o_d_e = torch.nan_to_num(emit_weight * f_d_e * mis_emit_weight, nan=0.0, posinf=0.0, neginf=0.0)
        t4 = time.time()

        rhs_color = L_o_d_b + L_o_d_e
        rhs_thp = self._rhs_thp.reshape(-1, self.dirs_per_point, 3)
        rhs_em = self._rhs_em.reshape(-1, self.dirs_per_point, 3)
        rhs_color = rhs_color * rhs_thp + rhs_em

        out_color = torch.mean(rhs_color, dim=1)
        dr.sync_device()

        out_color = torch.where(emitter_mask, self._emission, out_color)
        t5 = time.time()

        # print("##### Rendering time #####")
        # print("emission:", t1 - t0)
        # print("bsdf_emission:", t2 - t1)
        # print("bsdf_color:", t3 - t2)
        # print("emit_color:", t4 - t3)
        # print("out_color:", t5 - t4)
        
        return out_color

    def sample_lhs(self, seed : int = 0, pose : dict = None):
        if pose is not None:
            si = sample_si_pose(pose, self.scene, self.point_num, seed=seed)
        else:
            si = sample_si(self.scene, self.point_num, seed=seed)

        self.si_lhs = si
        self.si_rhs = self.get_rhs_from_lhs(si)
    
    def sample_bsdf_rhs(self, seed : int = 0):
        # Set sampler
        r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        r_sampler.seed(seed, self.point_num * self.dirs_per_point)

        ctx = mi.BSDFContext()
        bsdf_sample, bsdf_weight = self.si_rhs.bsdf().sample(
            ctx, self.si_rhs,
            r_sampler.next_1d(),
            r_sampler.next_2d(),
            active=True,
        )

        ray = self.si_rhs.spawn_ray(self.si_rhs.to_world(bsdf_sample.wo))
        si_bsdf_first = self.scene.ray_intersect(ray)
        si_bsdf, throughput, emission, _ = first_smooth(self.scene, r_sampler, ray, active=True)

        self.si_bsdf = si_bsdf
        self.si_bsdf_first = si_bsdf_first
        self._bsdf_thp = throughput.torch()
        self._bsdf_em = emission.torch()
        self._bsdf_sample = bsdf_sample
        self._bsdf_weight = bsdf_weight.torch()
        self._p_b_d_b = bsdf_sample.pdf

    def sample_emit_rhs(self, seed : int = 0, sub_emit : bool = False):
        # Set sampler
        r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        r_sampler.seed(seed, self.point_num * self.dirs_per_point)

        if sub_emit:
            pass
        else:
            active_em = mi.has_flag(self.si_rhs.bsdf().flags(), mi.BSDFFlags.Smooth)
            ds, em_weight = self.scene.sample_emitter_direction(
                self.si_rhs, r_sampler.next_2d(), True, active_em
            )
            # wo = self.si_rhs.to_local(ds.d)

        self._emit_sample = ds
        self._emit_weight = em_weight.torch()
        self._p_e_d_e = ds.pdf

    def eval_pdf(self):
        # Evaluate self emission
        self._emission = self.si_lhs.emitter(self.scene).eval(self.si_lhs).torch()

        ctx = mi.BSDFContext()

        # Evaluate p_b(d_e)
        wo_emit = self.si_rhs.to_local(self._emit_sample.d)
        bsdf_val, bsdf_pdf = self.si_rhs.bsdf().eval_pdf(
            ctx, self.si_rhs, wo_emit, active=True
        )

        # Evaluate p_e(d_b)
        # wo_bsdf = self.si_rhs.to_world(self._bsdf_sample.wo)
        si_rhs_delta = mi.has_flag(self._bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
        ds: mi.DirectionSample3f = mi.DirectionSample3f(
            self.scene, self.si_bsdf_first, self.si_rhs)
        em_pdf = mi.Float(0.0)
        em_pdf = self.scene.pdf_emitter_direction(
            self.si_rhs, ds, ~si_rhs_delta)
        
        self._p_b_d_e = bsdf_pdf
        self._p_e_d_b = em_pdf
        self._f_d_e = bsdf_val.torch()

        # Evaluate MIS weights
        assert self._p_b_d_b is not None
        assert self._p_e_d_e is not None

        self._mis_bsdf = dr.detach(dr.select(self._p_b_d_b > 0,
            self._p_b_d_b / (self._p_b_d_b + self._p_e_d_b), 0)).torch()
        self._mis_emit = dr.select(self._emit_sample.delta, 1.0,
            dr.detach(dr.select(self._p_e_d_e > 0,
            self._p_e_d_e / (self._p_e_d_e + self._p_b_d_e), 0))).torch()
    
    def get_rhs_from_lhs(self, si_lhs: mi.SurfaceInteraction3f, seed : int = 0):
        # Set sampler
        r_sampler: mi.Sampler = mi.load_dict({"type": "independent"})
        r_sampler.seed(seed, self.point_num * self.dirs_per_point)

        # Sample incident directions, trace intersection
        indices = dr.arange(mi.Int, 0, self.point_num)
        rhs_indices = dr.repeat(indices, self.dirs_per_point)
        si_rhs = dr.gather(mi.SurfaceInteraction3f, si_lhs, rhs_indices)

        # Intersect ray
        si_rhs, throughput, emission, _ = first_smooth(self.scene, r_sampler, si_rhs, active=True)

        self._rhs_thp = throughput.torch()
        self._rhs_em = emission.torch()

        return si_rhs
    
    def get_wr_itsc(self):
        # Trace secondary ray in the reflection direction, get intersection
        wi = self.si_lhs.to_world(self.si_lhs.wi)
        n = self.si_lhs.sh_frame.n
        wr = 2 * dr.dot(wi, n) * n - wi

        ray = mi.Ray3f(self.si_lhs.p + wr * 1e-5, wr)
        self.si_wr = self.scene.ray_intersect(ray)
    
    def to(self, device: str = "cuda", dtype=torch.float32):
        """
        Move all tensor attributes to the specified device
        """
        self._emission = self._emission.to(device=device, dtype=dtype)
        self._bsdf_thp = self._bsdf_thp.to(device=device, dtype=dtype)
        self._bsdf_em = self._bsdf_em.to(device=device, dtype=dtype)
        self._bsdf_weight = self._bsdf_weight.to(device=device, dtype=dtype)
        self._mis_bsdf = self._mis_bsdf.to(device=device, dtype=dtype)
        self._f_d_e = self._f_d_e.to(device=device, dtype=dtype)
        self._emit_weight = self._emit_weight.to(device=device, dtype=dtype)
        self._mis_emit = self._mis_emit.to(device=device, dtype=dtype)
        self._rhs_thp = self._rhs_thp.to(device=device, dtype=dtype)
        self._rhs_em = self._rhs_em.to(device=device, dtype=dtype)
    

def render_pt(scene: mi.Scene, sampler: mi.Sampler, si: mi.SurfaceInteraction3f):

    si: mi.SurfaceInteraction3f = mi.SurfaceInteraction3f(si)
    ray: mi.Ray3f = mi.Ray3f(si.p, si.to_world(si.wi))
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

        # si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

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

        si = scene.ray_intersect(ray, active_next)

    return dr.select(valid_ray, result, 0.0), valid_ray