import math
import numpy as np
from auxiliary import Object3D, Triangle3D, create_rigid_transformation, rtrans, vector_angle

def sort_by_x(vector: np.ndarray):
    return float(vector[0])
def sort_by_x_group(group: list[np.ndarray]):
    for item in group:
        if isinstance(item, np.ndarray):
            return float(group[0][0]) #return the x value of the first coordinate of a group
    raise ValueError


def generate_gcode(command_series:list[tuple[list, np.ndarray|None]]):
    """
    command(s) paired with a coordinate or in the same list are interpreted as simultaneous processes. standalone commands should be their own item in the command_series
    command format: (list of gcode commands, coordinate)\n
    command format: tuple(["M204", ";TYPE:Perimiter"] | "G1", None|ndarray[float(x)|None, float(y)|None, float(z)|None])
    command format: ";TYPE;Perimeter" or "M204" or
    (["G1", "F2.000"], np.ndarray)
    (["E2.420", ";Retract nozzle"], None)
    """
    lines = []
    for item in command_series:
        assert (len(item) == 2 and isinstance(item, tuple)), \
            f"Error! generate_gcode(): Unknown item in command_series: {item}, idx:{command_series.index(item)}"

        gcode_line = []
        coordinate = item[1]
        command_list = item[0]

        if not coordinate is None:
            for i, axis in zip([0,1,2],"XYZ"):
                if not coordinate[i] is None:
                    gcode_line.append(f"{axis}{coordinate[i]:.3f}")

        for command in command_list:
            if command[0] in "GMTD":
                gcode_line.insert(0, command)
            elif command[0] in "FE;": # "F232.110; E1.002; E12424"
                gcode_line.append(command)
            else:
                raise SyntaxError(f"Error! generate_gcode(): unknown command provided: {item}, idx: {command_series.index(item)}")

        lines.append(" ".join(gcode_line)+"\n")
    return lines


def get_min_max_y(object3d):
    mesh_vertex_ys = []
    for edge_y in object3d.mesh:
        y1 = object3d.vertices[edge_y[0]][1]
        y2 = object3d.vertices[edge_y[1]][1]
        mesh_vertex_ys.extend([y1, y2])
    return min(mesh_vertex_ys), max(mesh_vertex_ys)


def generate_toolpaths(objects: dict[int,Object3D]) -> list:
    """
    1. Create planes spaced at ironing pitch and at ironing angle
    2. For each plane, find all points of intersection with ironing-approved triangles
    remove duplicate points and order from one edge of the scan to the other.
    3. Should now have a list of points that trace the ironing path over an object. Now repeat for
    all objects.
    4. From lists of points, generate G-code that can be slapped on the end of the original gcode
    for cool non-planar ironing.
    """
    from nonplanar_iron import prefs

    gcode_lines = []

    # if prefs["fill_pattern"] == "linear":
    rotation_matrix = create_rigid_transformation(None,(0,0,prefs["scan_angle"]),None)

    for object3d in objects.values():
        # vertex_ids_list = []
        object3d.apply_transform(False, rotation_matrix)

        min_y, max_y = get_min_max_y(object3d)
        print(f"Slice range: Y=({min_y}, {max_y})")

        slice_planes = [i for i in np.arange(min_y + prefs["pass_spacing"] / 2, max_y, prefs["pass_spacing"])]
        print(f"{len(slice_planes)} slice planes generated")

        all_mesh_groups = []
        for group in object3d.face_groups:
            all_mesh_groups.append(object3d.parse_mesh(group))
        print(f"{len(all_mesh_groups)} mesh groups generated")

        point_series=[] #list[rows[face groups[coordinates]]]
        group_idx={} # {group id: [point rows belonging to group]}
        for i in range(len(slice_planes)):
            plane = slice_planes[i]
            point_row=[]
            edge_counter=[0,0,0]

            for a in range(len(all_mesh_groups)):
                mesh_group = all_mesh_groups[a]
                group_idx[a] = []
                point_row_group=[]
                for j in range(len(mesh_group)):
                    edge = mesh_group[j]
                    edge = (object3d.vertices[edge[0]], object3d.vertices[edge[1]])
                    if (edge[0][1]-plane) * (edge[1][1]-plane) > 0:
                        # Both endpoints lie on the same side of the plane, neither on the plane
                        # Add no points to the point series
                        continue

                    elif (edge[0][1]-plane) * (edge[1][1]-plane) < 0:
                        #True if endpoints are on opposing sides of the plane
                        #Add intersection of edge and plane to point series
                        A = np.array([[1, 0, edge[0][0] - edge[1][0]],
                                      [0, 1, edge[0][2] - edge[1][2]],
                                      [0, 0, edge[1][1] - edge[0][1]]])
                        b = np.array([edge[0][0], edge[0][2], plane - edge[0][1]])
                        result = np.linalg.solve(A,b)
                        point_row_group.append(np.array([result[0],plane,result[1]]))
                        edge_counter[0] += 1

                    elif np.isclose(edge[0][1]-plane, 0) and np.isclose(edge[1][1]-plane, 0):
                        #True if both endpoints lie on the plane
                        #Add both points to point series
                        point_row_group.append(edge[0])
                        point_row_group.append(edge[1])
                        edge_counter[1] += 1

                    elif np.isclose(edge[0][1]-plane * edge[1][1]-plane, 0):
                        #True if either endpoint lies on the plane
                        #Add the point that lies on the plane to the point series
                        if np.isclose(edge[0][1]-plane, 0):
                            point_row_group.append(edge[0])
                        elif np.isclose(edge[1][0]-plane, 0):
                            point_row_group.append(edge[1])
                        else:
                            input("Congrats, you somehow broke the code. See plane generation in generate_toolpaths")
                        edge_counter[2] += 1

                    else:
                        #Both endpoints lie on the same side of the plane, neither on the plane
                        #Add no points to the point series
                        pass

                print(f"object {object3d.id}: plane {i}/{len(slice_planes)}, Y={plane:.3f}: {edge_counter[0]+edge_counter[1]+edge_counter[2]} edges: {edge_counter[0]} pierce, {edge_counter[1]} coincident, {edge_counter[2]} on-surface: {len(point_row_group)} points added", end="\n")
                if point_row_group:
                    point_row.append(point_row_group)
                    group_idx[a].append(point_row_group)

            point_series.append(point_row)
        print(f"object {object3d.id}: _ points added to point series")

        #group_idx gets sorted???
        for i in range(len(point_series)):
            for j in range(len(point_series[i])):
                point_row_group = point_series[i][j]
                point_series[i][j] = sorted(point_series[i][j], key=sort_by_x)
            point_row = point_series[i]
            point_series[i] = sorted(point_series[i], key=sort_by_x_group)
        print(f"object {object3d.id}: point series rows sorted")

        print(group_idx[0])
        for i in range(len(point_series)):
            for j in range(len(point_series[i])):
                for k in range(len(point_series[i][j])):
                    point_series[i][j][k] = rtrans(rotation_matrix, point_series[i][j][k], inverse=True)
        print(f"object {object3d.id}: untransformed point series")
        input(group_idx[0])

        for i in range(len(point_series)):
            if i % 2:
                for j in range(len(point_series[i])):
                    point_series[i][j] = point_series[i][j][::-1]
                point_series[i] = point_series[i][::-1]
        print(f"object {object3d.id}: point series rows alternated")

        #dissolve all point row groups into rows, a single list
        for i in range(len(point_series)):
            new_series = []
            for j in range(len(point_series[i])):
                row_group = point_series[i][j]
                for k in range(len(point_series[i][j])):
                    if k == 0:
                        point_series[i][j][k] = (["G0"], point_series[i][j][k])
                    else:
                        point_series[i][j][k] = (["G1"], point_series[i][j][k])
                new_series.extend(point_series[i][j])
            point_series[i] = new_series

        # for i in range(len(point_series)):
        #     if i % 2:
        #         point_series[i] = point_series[i][::-1]
        # print(f"object {object3d.id}: point series rows alternated")

        point_series_consolidated = []
        for row in point_series:
            point_series_consolidated.extend(row)
        point_series = point_series_consolidated

        gcode = generate_gcode(point_series)
        print(f"object {object3d.id}: gcode generated")
        gcode_lines.extend(gcode)

    return gcode_lines
    print("writing to file")
    write_lines(sys.argv[2], gcode_lines)
    print(f"successfully wrote: {sys.argv[2]}")
