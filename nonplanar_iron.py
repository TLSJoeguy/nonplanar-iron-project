import math, sys
import matplotlib.pyplot as plt

import numpy as np
from auxiliary import Object3D, Triangle3D, create_rigid_transformation, rtrans, vector_angle, prefs, get_config_data, read_lines, write_lines
from parser_3mf import parse_3mf
import gcode_generation

MIN_Z_TOLERANCE = 0.01 #mm
sys.setrecursionlimit(1000000)


def plot_triangle_z(tri: Triangle3D):
    x_data = []
    y_data = []
    for v in tri.vertices():
        x_data.append(v[0])
        y_data.append(v[1])
    x_data.append(x_data[0])
    y_data.append(y_data[0])

    plt.plot(x_data, y_data)
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.draw()
    plt.pause(0.01)


def evaluate_tri_normals(objects:dict[int,Object3D]):
    for key, item in list(objects.items()):
        selected_tris = []
        for tri in item.tris:
            user_ang = np.pi/2 - prefs["nozzle_angle"]
            proj_user_ang = abs(np.cos(user_ang))
            proj_tri_ang = np.linalg.norm(np.dot(tri.normal(), np.array([1, 1, 0])))
            if proj_tri_ang <= proj_user_ang and tri.normal()[2] > 0 and prefs["require_custom_supports"] and tri.support == "8":
                selected_tris.append(tri)
            elif proj_tri_ang <= proj_user_ang and tri.normal()[2] > 0 and not prefs["require_custom_supports"]:
                selected_tris.append(tri)
            # else:
            #     print("not")
        print(f"{item.name}: kept {len(selected_tris)} out of {len(item.tris)} tris.")
        item.tris = selected_tris
        if len(selected_tris) == 0:
            removed = objects.pop(key, None)
            print(f"removed {removed.name}")


def group_consecutive_faces(objects: dict[int,Object3D]):
    for key, object3d in objects.items():
        object_faces = []
        seen_tris = []

        def evaluate_edge(tri: Triangle3D):
            seen_tris.append(tri)
            investigation = [tri]
            for tri_2 in tri.neighbors():
                if not tri_2 is None and not tri_2 in seen_tris and vector_angle(tri.normal(), tri_2.normal()) < prefs["face_tolerance"]:
                     investigation.extend(evaluate_edge(tri_2))

            return investigation

        while len(seen_tris) < len(object3d.tris):
            for tri in object3d.tris:
                if tri not in seen_tris:
                    object_faces.append(evaluate_edge(tri))

        object3d.face_groups = object_faces
        print(f"object {object3d.id}: {len(object_faces)} face groups found")


def main(infile, outfile):
    objects = parse_3mf(infile)
    for object3d in objects.values():
        object3d.apply_transform()
    evaluate_tri_normals(objects)
    for object3d in objects.values():
        object3d.parse_tri_edges()
        object3d.parse_mesh()
    group_consecutive_faces(objects)
    gcode_lines = gcode_generation.generate_toolpaths(objects)
    write_lines(outfile, gcode_lines)
    print(f"successfully wrote: {sys.argv[2]}")


if __name__ == "__main__":
    config_file = "nonplanar_iron.config"
    prefs = get_config_data(config_file)
    main(sys.argv[1],sys.argv[2])