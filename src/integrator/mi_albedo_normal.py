import mitsuba as mi

from typing import List
import drjit as dr
from mitsuba import Integrator


class BAlbedoNormalIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

    def aov_names(self: Integrator) -> List[str]:
        return super().aov_names() + ["sh_normal.x", "sh_normal.y", "sh_normal.z"]

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        bsdf: mi.BSDF = si.bsdf(ray)

        valid = active & si.is_valid()
        # albedo: [0, 1]
        albedo = dr.select(valid, bsdf.eval_diffuse_reflectance(si, valid), mi.Color3f(0.0, 0.0, 0.0))
        # sh_normal: [-1, 1]
        sh_normal = dr.select(valid, si.sh_frame.n, mi.Color3f(0.0, 0.0, 0.0))
        return albedo, si.is_valid(), [sh_normal.x, sh_normal.y, sh_normal.z]


mi.register_integrator("b_albedo_normal", lambda props: BAlbedoNormalIntegrator(props))


class BAlbedoNormalDepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

    def aov_names(self: Integrator) -> List[str]:
        return super().aov_names() + ["sh_normal.x", "sh_normal.y", "sh_normal.z", "depth"]

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        bsdf: mi.BSDF = si.bsdf(ray)

        valid = active & si.is_valid()
        # albedo: [0, 1]
        albedo = dr.select(valid, bsdf.eval_diffuse_reflectance(si, valid), mi.Color3f(0.0, 0.0, 0.0))
        # sh_normal: [-1, 1]
        sh_normal = dr.select(valid, si.sh_frame.n, mi.Color3f(0.0, 0.0, 0.0))
        # depth: [0, inf]
        depth = dr.select(valid, si.t, 0.0)
        return albedo, si.is_valid(), [sh_normal.x, sh_normal.y, sh_normal.z, depth]


mi.register_integrator("b_albedo_normal_depth", lambda props: BAlbedoNormalDepthIntegrator(props))


class BAlbedoDepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

    def aov_names(self: Integrator) -> List[str]:
        return super().aov_names() + ["depth"]

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active)
        bsdf: mi.BSDF = si.bsdf(ray)

        valid = active & si.is_valid()
        # albedo: [0, 1]
        albedo = dr.select(valid, bsdf.eval_diffuse_reflectance(si, valid), mi.Color3f(0.0, 0.0, 0.0))
        # depth: [0, inf]
        depth = dr.select(valid, si.t, 0.0)
        return albedo, si.is_valid(), [depth]


mi.register_integrator("b_albedo_depth", lambda props: BAlbedoDepthIntegrator(props))
