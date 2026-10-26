# from sect.triangulation import Triangulation
# from sect.triangulation import constrained_delaunay_triangles
# import numpy as np

# def polygon2mesh(polygon):#,mesh_vertices,mesh_faces,mesh_vertices_id,mesh_vertices_color,color=[0,0,0]):
#     polygon = [(x,y) for x,y in polygon]
#     tri = np.array(constrained_delaunay_triangles(polygon))
#     mesh_vertice = tri.reshape(tri.shape[0] * 3, 2)
#     mesh_vertice = np.concatenate((mesh_vertice,np.zeros(mesh_vertice.shape[0]).reshape(-1,1)),1)
#     mesh_face = np.arange(mesh_vertice.shape[0]).reshape(tri.shape[0], 3)
#     return mesh_vertice,mesh_face#,mesh_vertices_id,mesh_vertices_color

import numpy as np
from ground.base import get_context
from sect.triangulation import Triangulation

# Initialize geometry context and types
context = get_context()
Point = context.point_cls
Contour = context.contour_cls
Polygon = context.polygon_cls

def polygon2mesh(polygon):
    exterior = Contour([Point(x, y) for x, y in polygon])
    poly = Polygon(exterior, holes=[])
    triangles = Triangulation.constrained_delaunay(poly, context=context).triangles()

    # Just flatten the vertices from triangles in order
    mesh_vertices = []
    mesh_faces = []

    for i, tri in enumerate(triangles):
        pts = [(v.x, v.y) for v in tri.vertices]
        face_indices = list(range(len(mesh_vertices), len(mesh_vertices) + 3))
        mesh_vertices.extend(pts)
        mesh_faces.append(face_indices)

    mesh_vertices = np.array(mesh_vertices)
    mesh_vertices = np.hstack([mesh_vertices, np.zeros((mesh_vertices.shape[0], 1))])  # add z=0
    mesh_faces = np.array(mesh_faces)

    return mesh_vertices, mesh_faces