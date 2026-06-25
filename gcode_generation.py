import math
import numpy as np
from auxiliary import Object3D, Triangle3D, create_rigid_transformation, rtrans, vector_angle

def sort_by_x(vector: np.ndarray):
    return float(vector[0])


def sort_by_y_groups(group_rows):
    print(group_rows)
    return float(group_rows[0][0][1])


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


def sort_group_rows(face_groups):
    for idx, group_rows in face_groups.items():
        new_rows = []
        for i, row in enumerate(group_rows):
            if i % 2:
                row = sorted(row, key=sort_by_x, reverse=True)
            else:
                row = sorted(row, key=sort_by_x)

            new_rows.append(row)
        face_groups[idx] = new_rows
    return face_groups


def untransform_face_groups(face_groups, rotation_matrix):
    for idx, group_rows in face_groups.items():
        new_rows = []
        for row in group_rows:
            new_row = []
            for coordinate in row:
                new_row.append(rtrans(rotation_matrix, coordinate, inverse=True))
            new_rows.append(new_row)
        face_groups[idx] = new_rows
    return face_groups


def append_gcode_commands(face_groups):
    for idx, group in face_groups.items():
        new_rows = []
        for row in group:
            row = [(["G0"], coordinate) if i == 0 else (["G1"], coordinate) for i, coordinate in enumerate(row)]
            new_rows.append(row)
        face_groups[idx] = new_rows
    return face_groups


def add_group_outline(face_groups, mesh_boundaries, object3d):
    for i, boundary_edges in enumerate(mesh_boundaries):
        orig_len = len(boundary_edges)
        all_boundary_loops = []
        ordered_edges = [boundary_edges[0][0]]
        while boundary_edges:
            for edge in boundary_edges:
                if edge[0] == ordered_edges[-1]:
                    ordered_edges.append(edge[-1])
                    boundary_edges.remove(edge)
                elif edge[-1] == ordered_edges[-1]:
                    ordered_edges.append(edge[0])
                    boundary_edges.remove(edge)

            if ordered_edges[0] == ordered_edges[-1]:
                all_boundary_loops.append(ordered_edges.copy())
                ordered_edges = [boundary_edges[0][0]] if boundary_edges else None

        # assert orig_len == len(ordered_edges)-1, f"Error! add_group_outline: len(boundary_edges)={orig_len}, len(ordered_edges)={len(ordered_edges)}"
        for j, loop in enumerate(all_boundary_loops):
            all_boundary_loops[j] = [object3d.vertices[a] for a in loop]
        print(f"group {i}: {len(all_boundary_loops)} boundaries")
        mesh_boundaries[i] = all_boundary_loops
            
    
    for idx, group_rows in face_groups.items():
        face_groups[idx].extend(mesh_boundaries[idx])

    return face_groups


def consolidate_groups(face_groups):
    series = []
    for idx, group_rows in face_groups.items():
        for row in group_rows:
            series.extend(row)
    return series


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
        all_mesh_boundaries = []
        for group in object3d.face_groups:
            all_mesh_groups.append(object3d.parse_mesh(group))
            all_mesh_boundaries.append(object3d.parse_mesh(group, boundaries_only=True))
        print(f"{len(all_mesh_groups)} mesh groups generated")
        print(f"{len(all_mesh_boundaries)} mesh boundaries generated")

        face_groups = {a: [] for a in range(len(all_mesh_groups))} # {group id: [point rows belonging to group]}
        for i in range(len(slice_planes)):
            plane = slice_planes[i]

            for a, mesh_group in enumerate(all_mesh_groups):
                point_row=[]
                edge_counter = [0, 0, 0]
                for j, edge in enumerate(mesh_group):
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
                        point_row.append(np.array([result[0],plane,result[1]]))
                        edge_counter[0] += 1

                    elif np.isclose(edge[0][1]-plane, 0) and np.isclose(edge[1][1]-plane, 0):
                        #True if both endpoints lie on the plane
                        #Add both points to point series
                        point_row.append(edge[0])
                        point_row.append(edge[1])
                        edge_counter[1] += 1

                    elif np.isclose(edge[0][1]-plane * edge[1][1]-plane, 0):
                        #True if either endpoint lies on the plane
                        #Add the point that lies on the plane to the point series
                        if np.isclose(edge[0][1]-plane, 0):
                            point_row.append(edge[0])
                        elif np.isclose(edge[1][0]-plane, 0):
                            point_row.append(edge[1])
                        else:
                            input("Congrats, you somehow broke the code. See plane generation in generate_toolpaths")
                        edge_counter[2] += 1

                    else:
                        #Both endpoints lie on the same side of the plane, neither on the plane
                        #Add no points to the point series
                        pass

                # print(f"object {object3d.id}: plane {i}/{len(slice_planes)}, Y={plane:.3f}, group={a}/{len(all_mesh_groups)}: {edge_counter[0]+edge_counter[1]+edge_counter[2]} edges: {edge_counter[0]} pierce, {edge_counter[1]} coincident, {edge_counter[2]} on-surface: {len(point_row)} points added", end="\n")
                if point_row:
                    face_groups[a].append(point_row)

        print(f"object {object3d.id}: {len(face_groups)} face groups sliced")

        face_groups = sort_group_rows(face_groups)
        print(f"object {object3d.id}: all group_rows sorted and alternated")
        
        face_groups = add_group_outline(face_groups, all_mesh_boundaries, object3d)
        print(f"object {object3d.id}: added group outlines")

        face_groups = sorted(list(face_groups.values()), key=sort_by_y_groups)
        face_groups = {i: value for i, value in enumerate(face_groups)}
        print(f"object {object3d.id}: sorted groups by min y coords")

        face_groups = untransform_face_groups(face_groups, rotation_matrix)
        print(f"object {object3d.id}: untransformed point series")

        face_groups = append_gcode_commands(face_groups)
        print(f"object {object3d.id}: added gcode commands to coordinates")

        command_series = consolidate_groups(face_groups)
        print(f"object {object3d.id}: consolidated all face groups into stream of commands")

        gcode = generate_gcode(command_series)
        print(f"object {object3d.id}: gcode generated")

        gcode_lines.extend(gcode)

    return gcode_lines
    print("writing to file")
    write_lines(sys.argv[2], gcode_lines)
    print(f"successfully wrote: {sys.argv[2]}")
