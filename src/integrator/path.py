
from src.sample.path import render_pt

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


class PTIntegrator(mi.SamplingIntegrator):

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.max_depth = props.get("max_depth", 16)

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        ray = mi.Ray3f(ray)
        si = scene.ray_intersect(ray, active)
        L, valid = render_pt(scene, sampler, si)
        
        return L, valid, []


mi.register_integrator("pt", lambda props: PTIntegrator(props))