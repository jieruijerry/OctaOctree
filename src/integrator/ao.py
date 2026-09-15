import torch

from src.sample.lhs_rhs import first_smooth

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def render_rtao(
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.RayDifferential3f,
    radius: float = 0.1,
    active: bool = True,
):
    """
    Render ray-traced ambient occlusion
    """
    si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

    ao_wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
    ao_wi = si.to_world(ao_wi)
    ao_ray = mi.Ray3f(si.p, ao_wi)
    ao_ray.maxt = mi.Float(radius)

    visible = ~scene.ray_test(ao_ray, active)

    result = dr.select(
        visible,
        mi.Color3f(1.0),
        mi.Color3f(0.0)
    )

    return result, si.is_valid()

def render_di(
    scene: mi.Scene,
    sampler: mi.Sampler,
    ray: mi.RayDifferential3f,
    active: bool = True,
):
    """
    Render direct illumination
    """
    si, throughput, emission, _ = first_smooth(scene, sampler, ray, active)
    bsdf: mi.BSDF = si.bsdf()
    ctx: mi.BSDFContext = mi.BSDFContext()

    ds, em_weight = scene.sample_emitter_direction(
        si, sampler.next_2d(), True, active
    )
    wo = si.to_local(ds.d)
    bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active)

    return dr.select(emission > 0.0, emission, em_weight * bsdf_val * throughput)


class RTAOIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.radius = props.get("radius", 0.1)
    
    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:
        
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)

        di = render_di(scene, sampler, ray, active)
        ao, valid = render_rtao(scene, sampler, ray, self.radius, active)

        return 2 * di + 0.2 * ao * si.bsdf(ray).eval_diffuse_reflectance(si), valid, []


mi.register_integrator("ao", lambda props: RTAOIntegrator(props))