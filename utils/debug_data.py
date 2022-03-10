import json
import time
import os
import numpy as np
import open3d
from open3d.cpu.pybind.geometry import PointCloud, LineSet
from open3d.cpu.pybind.utility import Vector3dVector, Vector2iVector
from open3d.cpu.pybind.visualization import Visualizer

if __name__ == "__main__":

    with open('data' + os.sep + 'dataset.txt', 'r') as f:
        dataset = json.load(f)

    with open('data' + os.sep + 'classes.txt', 'r') as f:
        classes = json.load(f)

    with open('assets' + os.sep + 'edges.txt', 'r') as f:
        edges = json.load(f)
    edges = np.array(edges)

    i = 0


    def sample_spherical(npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec.swapaxes(0, 1)
        return vec


    flag = True
    pc = PointCloud()
    lines = LineSet()
    coord = open3d.cpu.pybind.geometry.TriangleMesh.create_coordinate_frame()

    sphere_points = sample_spherical(500)
    sphere = PointCloud()
    sphere.points = Vector3dVector(sphere_points)

    vis = Visualizer()

    for x, y in dataset:

        y = np.array(y)
        max_index = np.where(y == np.amax(y))[0]
        action = classes[int(max_index)]

        vis.create_window(window_name=action)

        x = np.array(x)  # from list of list to numpy array
        x = x.reshape(x.shape[0], -1, 3)  # restore 3d coordinates

        for pose in x:
            lines.points = Vector3dVector(pose)
            lines.lines = Vector2iVector(edges)

            aux = PointCloud()
            aux.points = Vector3dVector(pose)
            pc.clear()
            pc += aux

            if flag:
                vis.add_geometry(pc)
                vis.add_geometry(lines)
                # vis.add_geometry(coord)
                vis.add_geometry(sphere)
                flag = False

            vis.update_geometry(pc)
            vis.update_geometry(lines)
            # vis.update_geometry(coord)
            vis.update_geometry(sphere)
            vis.poll_events()
            vis.update_renderer()

            time.sleep(0.1)

        time.sleep(0.3)

        vis.close()
