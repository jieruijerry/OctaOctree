import time
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


class DepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.color = props.get("color", mi.Color3f(0.9, 0.9, 0.9))
        self.background = props.get("background", mi.Color3f(0.1))
        self.ray_type = props.get("ray_type", "primary")

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Compute max depth of the scene

        camera: mi.ProjectiveCamera = scene.sensors()[0]
        camera_pos: mi.Point3f = camera.world_transform().translation()
        box: mi.BoundingBox3f = scene.bbox()

        max_depth: mi.Float = mi.Float(0.0)
        for i in range(8):
            max_depth = dr.maximum(
                max_depth,
                dr.norm(box.corner(i) - camera_pos)
            )

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        wi = si.to_world(si.wi)
        n = si.sh_frame.n
        wr = 2 * dr.dot(wi, n) * n - wi

        secondary_ray = mi.Ray3f(si.p, wr)
        scd_si = scene.ray_intersect(secondary_ray)

        # Render depth map
        result = mi.Color3f(0.0)
        if self.ray_type == "primary":
            dist = dr.norm(si.p - camera_pos) / max_depth
        elif self.ray_type == "secondary":
            dist = dr.norm(scd_si.p - si.p) / max_depth
        else:
            raise ValueError("Invalid ray type:", self.ray_type)
        result[si.is_valid()] = self.color * (1.0 - dist)

        return result, si.is_valid(), []


class MCDepthIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.color = props.get("color", mi.Color3f(0.9, 0.9, 0.9))
        self.background = props.get("background", mi.Color3f(0.1))
        self.ray_type = props.get("ray_type", "primary")

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:
        
        camera: mi.ProjectiveCamera = scene.sensors()[0]
        camera_pos: mi.Point3f = camera.world_transform().translation()
        box: mi.BoundingBox3f = scene.bbox()

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        ctx = mi.BSDFContext()
        bsdf_sample, bsdf_weight = si.bsdf().sample(
            ctx, si,
            sampler.next_1d(),
            sampler.next_2d(),
            active,
        )

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        si_scd = scene.ray_intersect(ray)

        max_depth: mi.Float = mi.Float(3.0)

        # Render depth map
        result = mi.Color3f(0.0)
        if self.ray_type == "primary":
            dist = si.t / max_depth
        elif self.ray_type == "secondary":
            dist = si_scd.t / max_depth
        else:
            raise ValueError("Invalid ray type:", self.ray_type)
        result[si.is_valid()] = self.color * (1 - dist)

        return result, si.is_valid(), []


class AlbedoIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.specular = False

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Render albedo

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        if self.specular:
            color = si.bsdf(ray).eval_specular_reflectance(si)
        else:
            color = si.bsdf(ray).eval_diffuse_reflectance(si)

        result = mi.Color3f(0.0)
        result[si.is_valid()] = color

        return result, si.is_valid(), []


class NormalIntegrator(mi.SamplingIntegrator):
    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.background = props.get("background", mi.Color3f(0.1))

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True
    ) -> tuple[mi.Color3f, bool, list[float]]:

        # Render normal map

        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

        result = mi.Color3f(0.0)
        result[si.is_valid()] = dr.fma(si.n, 0.5, 0.5)
        result[~si.is_valid()] = self.background

        return result, si.is_valid(), []


mi.register_integrator("depth", lambda props: DepthIntegrator(props))
mi.register_integrator("mc_depth", lambda props: MCDepthIntegrator(props))
mi.register_integrator("albedo", lambda props: AlbedoIntegrator(props))
mi.register_integrator("normal", lambda props: NormalIntegrator(props))
