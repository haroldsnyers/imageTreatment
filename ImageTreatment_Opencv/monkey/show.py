import json
import numpy as np
from mayavi import mlab

points = json.loads(open("./point.txt", "r").read())

x = points["x"]
y = points["y"]
z = points["z"]

x = np.array(x)
y = np.array(y)
z = np.array(z)

# mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(400, 350))
# bgcolor : background color
# fgcolor : foreground color (text and others)
mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

# Visualize the points
# points3d -> plots glyphs (like points) at the position of the supplied data
# first 3 arrays : define points in 3D space
# Fourth array -> associated scalar value for each point -> to modulate
#                 the color (based on Z coordinate and the size of the points
pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=100)

# Create and visualize the mesh

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)
# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

mlab.view(0, 2000, -2500)

mlab.show()