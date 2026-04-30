#%%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    origins = t.zeros(num_pixels, 3)
    ends = t.cartesian_prod(t.tensor([1.]), t.linspace(-y_limit, y_limit, num_pixels), t.tensor([0.]))
    return t.stack([origins, ends], dim=-2)


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %%

fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)
# %%
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray
    L1, L2 = segment

    # solve the intersection between the ray 0-D-> and the line <-L1-L2->
    A = einops.rearrange([D, L1-L2], 'h w -> w h')
    B = einops.rearrange(L1 - O, 'w -> w 1')

    result = t.linalg.lstsq(A, B)
    x = result.solution

    residual = A @ x - B
    if not t.allclose(residual, t.zeros_like(residual), atol=1e-3):
        return False # no solution, lines don't intersect
    u, v = x
    if u < 0 or v < 0 or v > 1:
        return False
    return True

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    nrays = rays.shape[0]
    nsegments = segments.shape[0]

    rays = einops.repeat(rays, 'b i c -> (nsegments b) i c', nsegments=nsegments)[..., :2] # 1 2 3 1 2 3 1 2 3 1 2 3
    segments = einops.repeat(segments, 'b i c -> (b nrays) i c', nrays=nrays)[..., :2] # 1 1 1 2 2 2 3 3 3 4 4 4
    O = rays[:, 0, :]
    D = rays[:, 1, :]
    L1 = segments[:, 0, :]
    L2 = segments[:, 1, :]

    A = einops.rearrange([D, L1-L2], 'i b c -> b c i') # batch x ndim x 2
    B = einops.rearrange(L1 - O, 'b w -> b w 1') # batch x ndim x 1
    
    result = t.linalg.lstsq(A, B)
    x = result.solution # batch x 2 x 1
    
    # A @ x - B on each batch
    residual = einops.einsum(A, x, 'b i k, b k j -> b i j') - B
    # it's a hit if the residual is small, u > 0, and v in [0, 1]
    low_residual = t.all(t.isclose(residual, t.zeros_like(residual), atol=1e-3), dim=1)
    intersects = low_residual & (x[:, 0, :] >= 0) & (x[:, 1, :] >= 0) & (x[:, 1, :] <= 1)
    return t.any(einops.rearrange(intersects, '(nsegments nrays) 1 -> nsegments nrays', nsegments=nsegments), dim=0)


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    """
    For each ray, return True if it intersects any segment.
    """
    ys = t.linspace(-y_limit, y_limit, num_pixels_y)
    zs = t.linspace(-z_limit, z_limit, num_pixels_z)
    grid_y, grid_z = t.meshgrid(ys, zs)
    yz_tuples = einops.rearrange([grid_y, grid_z], 'coord y z -> (y z) coord')

    rays = t.zeros(num_pixels_y*num_pixels_z, 2, 3)
    # rays[:, 0, :] are all (0, 0, 0)
    # rays[:, 1, :] are (1, y, z)
    rays[:, 1, 0] = 1
    rays[:, 1, 1:] = yz_tuples
    return rays


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)

# %%
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack([-D, B-A, C-A], dim=1)
    out = t.unsqueeze(O-A, dim=1)

    result = t.linalg.lstsq(mat, out)
    x = result.solution
    residual = mat @ x - out
    if not t.allclose(residual, t.zeros_like(residual), atol=1e-3):
        return False
    s, u, v = x
    return s > 0 and u+v <= 1 and u >= 0 and v >= 0


tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]

    O = rays[:, 0, :] # nrays, 3
    D = rays[:, 1, :] # nrays, 3
    A, B, C = einops.repeat(triangle, 'pt c -> pt nrays c', nrays = nrays) # each is nrays, 3

    assert A.shape == (nrays, 3)
    assert O.shape == (nrays, 3)

    mats: Float[Tensor, "nrays 3 3"] = t.stack([-D, B-A, C-A], dim=-1)
    outs: Float[Tensor, "nrays 3 1"] = t.unsqueeze(O-A, dim=-1)

    assert mats.shape == (nrays, 3, 3)
    assert outs.shape == (nrays, 3, 1)

    dets: Float[Tensor, "nrays"] = t.linalg.det(mats)
    is_singular = dets.abs() < 1e-8
    mats[is_singular] = t.eye(3)

    sol: Float[Tensor, "nrays 3"] = t.linalg.solve(mats, outs).squeeze(-1)
    s, u, v = sol[:, 0], sol[:, 1], sol[:, 2]
    return ~is_singular & (s >= 0) & (u+v <= 1) & (u >= 0) & (v >= 0)


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # make everything (nrays x ntriangles x ...)
    rays = einops.repeat(rays, 'nrays rayPoints dims -> nrays ntriangles rayPoints dims', ntriangles=ntriangles)
    triangles = einops.repeat(triangles, 'ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims', nrays=nrays)

    O, D = t.unbind(rays, 2)
    A, B, C = t.unbind(triangles, 2)

    assert O.shape == (nrays, ntriangles, 3)
    assert A.shape == (nrays, ntriangles, 3)

    mat: Float[Tensor, "nrays ntriangles 3 3"] = t.stack([-D, B-A, C-A], dim=-1)
    out: Float[Tensor, "nrays ntriangles 3 1"] = (O-A).unsqueeze(dim=-1)

    assert mat.shape == (nrays, ntriangles, 3, 3)
    assert out.shape == (nrays, ntriangles, 3, 1)

    # mask singular matrices with identity to ensure all matrix equations are solvable. We will remember this later
    is_singular = t.det(mat).abs() < 1e-8
    mat[is_singular] = t.eye(3)

    sol: Float[Tensor, "nrays ntriangles 3"] = t.linalg.solve(mat, out).squeeze(-1)
    s, u, v = sol.unbind(-1)
    # matrix wasn't singular or intersection pt between ray and plane is outside traingle
    intersects = ~is_singular & (s >= 0) & (u+v <= 1) & (u >= 0) & (v >= 0)

    # x-coord where each ray and triangle intersects
    intersect_pt = O+D*(s.unsqueeze(-1))
    intersect_x = intersect_pt[..., 0]
    # mask non-intersecting points (singular matrix, intersects outside triangle) with infinity
    intersect_x[~intersects] = t.inf

    return einops.reduce(intersect_x, 'nrays ntriangles -> nrays', 'min')


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor([
        [t.cos(theta), 0, t.sin(theta)],
        [0, 1, 0],
        [-t.sin(theta), 0, t.cos(theta)]
    ])

# (1 0 0) -> (costheta  0 sintheta)
# (0 1 0) -> (0 1 0)
# (0 0 1) -> (-sintheta  0 costheta)

tests.test_rotation_matrix(rotation_matrix)
# %%
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.mps.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %%
def raytrace_mesh_lambert(
    rays: Float[Tensor, "nrays points=2 dims=3"],
    triangles: Float[Tensor, "ntriangles points=3 dims=3"],
    light: Float[Tensor, "dims=3"],
    ambient_intensity: float,
    device: str = "mps",
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if
    no intersection).

    Args:
        rays:   A tensor of rays, with shape `[nrays, 2, 3]`.
        triangles:  A tensor of triangles, with shape `[ntriangles, 3, 3]`.
        light:  A tensor representing the light vector, with shape `[3]`. We compute the intensity
                as the dot product of the triangle normals & the light vector, then set it to be
                zero if the sign is negative.
        ambient_intensity:  A float representing the ambient intensity. This is the minimum
                            brightness for a triangle, to differentiate it from the black background
                            (rays that don't hit any triangle).
        device: The device to perform the computation on.

    Returns:
        A tensor of intensities for each of the rays, flattened over the [y, z] dimensions. The
        values are zero when there is no intersection, and `ambient_intensity + intensity` when
        there is an interesection (where `intensity` is the dot product of the triangle's normal
        vector and the light vector, truncated at zero).
    """
    print('computing ray trace lambert')

    ntriangles = triangles.shape[0]
    nrays = rays.shape[0]

    # compute lightedness of each triangle
    normal_vecs: Float[Tensor, "ntriangles 3"] = t.cross(triangles[:, 2] - triangles[:, 0], triangles[:, 1] - triangles[:, 0], dim=1)
    normal_vecs = normal_vecs / normal_vecs.norm(dim=1, keepdim=True)
    triangle_intensity = t.linalg.vecdot(light, normal_vecs, dim=-1)
    # negative intensity means the traingle is facing the other way; these get clamped to ambient_intensity
    triangle_intensity = t.maximum(triangle_intensity, t.zeros_like(triangle_intensity)) + ambient_intensity

    # compute ray-triangle intersections, recycling code from earlier
    triangles = einops.repeat(triangles, 'ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims', nrays=nrays)
    rays = einops.repeat(rays, 'nrays rayPoints dims -> nrays ntriangles rayPoints dims', ntriangles=ntriangles)

    O, D = t.unbind(rays, 2)
    A, B, C = t.unbind(triangles, 2)

    assert O.shape == (nrays, ntriangles, 3)
    assert A.shape == (nrays, ntriangles, 3)

    mat: Float[Tensor, "nrays ntriangels 3 3"] = t.stack([-D, B-A, C-A], dim=-1)
    out: Float[Tensor, "nrays ntriangles 3"] = O-A

    # mask singular matrices with identity to ensure all matrix equations are solvable. We will remember this later
    is_singular = t.det(mat).abs() < 1e-8
    mat[is_singular] = t.eye(3)

    sol: Float[Tensor, "nrays ntriangles 3"] = t.linalg.solve(mat, out)
    s, u, v = sol.unbind(-1)
    # matrix wasn't singular or intersection pt between ray and plane is outside traingle
    intersects = ~is_singular & (s >= 0) & (u+v <= 1) & (u >= 0) & (v >= 0)

    # x-coord where each ray and triangle intersects
    intersect_pt: Float[Tensor, "nrays ntriangles 3"] = 0+D*(s.unsqueeze(-1))
    intersect_x: Float[Tensor, "nrays ntriangles"] = intersect_pt[..., 0]
    # mask non-intersecting points (singular matrix, intersects outside triangle) with infinity
    intersect_x[~intersects] = t.inf

    closest_x_intersect, closest_triangle = t.min(intersect_x, dim=-1)

    ray_lights = triangle_intensity[closest_triangle]
    ray_lights[~t.isfinite(closest_x_intersect)] = 0
    print('computed')

    return ray_lights

    
def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is the lighting intensity based on the angle of light & the surface of the triangle
    which this ray hits first.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)").show()


ambient_intensity = 0.5
light = t.tensor([0.0, -1.0, 1.0])
raytrace_function = partial(
    raytrace_mesh_lambert,
    ambient_intensity=ambient_intensity,
    light=light,
)

intensity = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_function, num_frames)
intensity = einops.rearrange(intensity, "frames (y z) -> frames y z", y=num_pixels_y)
display_video_with_lighting(intensity)
# %%
print(intensity.shape)
# %%
