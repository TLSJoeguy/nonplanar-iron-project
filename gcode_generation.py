import numpy as np
from auxiliary import Object3D, Triangle3D, GcodePoint, create_rigid_transformation, rtrans, vector_angle, len_nested


def sort_by_component(vector: np.ndarray | GcodePoint, component:str= "x"):
    """use as a key for .sort(); sorts a list of 1D numerical arrays by their first element"""
    comps = "xyz"
    assert component in comps, "Error! sort_by_x component must be one of the following: 'x', 'y', 'z'"

    if isinstance(vector, GcodePoint):
        vector = vector.coordinate
    return float(vector[comps.index(component)])

def sort_by_component_group(group: list[np.ndarray | GcodePoint], component:str= "x", sort_by_last_vector=False):
    """use as a key for .sort(); sorts a list of lists of 1D numerical arrays by the first element of the first array of each list"""
    comps = "xyz"
    assert component in comps, "Error! sort_by_x component must be one of the following: 'x', 'y', 'z'"

    vector = group[-1] if sort_by_last_vector else group[0]
    if isinstance(vector, GcodePoint):
        vector = vector.coordinate
    return float(vector[comps.index(component)])


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


def untransform_face_groups(points:list[list[list[GcodePoint]]], rotation_matrix):
    """additionally turns all GcodePoints into ndarrays"""
    new_points = []
    for idx, row in enumerate(points):
        new_row = []
        for sub_group in row:
            new_sub_group = []
            for coord in sub_group:
                if isinstance(coord, GcodePoint):
                    coord = coord.coordinate
                new_sub_group.append(rtrans(rotation_matrix, coord, inverse=True))
            new_row.append(new_sub_group)
        new_points.append(new_row)
    return new_points


def append_gcode_commands(points:list[list[list[GcodePoint]]]) -> list[list[list[tuple[list[str], np.ndarray]]]]:
    new_points = []
    for idx, row in enumerate(points):
        new_row = []
        for sub_group in row:
            sub_group = [(["G0"], coordinate) if i == 0 else (["G1"], coordinate) for i, coordinate in enumerate(sub_group)] #the first coordinate of each sub_row_group is G0
            new_row.append(sub_group)
        new_points.append(new_row)
    return new_points


def add_group_outline(gcode_points: list, mesh_boundaries:list[list[tuple[int,int]]], object3d):
    """orders mesh boundaries into continuous loops then tacks it onto the end of a gcode_points row list"""
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

        assert (orig_len == len_nested(all_boundary_loops)-2 and len(all_boundary_loops) == 2) or (orig_len == len_nested(all_boundary_loops)-1 and len(all_boundary_loops) == 1), f"Error! add_group_outline: len(boundary_edges)={orig_len}, len(ordered_edges)={len(ordered_edges)}"
        for j, loop in enumerate(all_boundary_loops):
            all_boundary_loops[j] = [object3d.vertices[a] for a in loop]
        print(f"group {i}: {len(all_boundary_loops)} boundaries")
        mesh_boundaries[i] = all_boundary_loops
            
    
    gcode_points.extend(mesh_boundaries)
    return gcode_points


def consolidate_groups(points: list[list[list[tuple[list[str], np.ndarray]]]]):
    series = []
    for row in points:
        for sub_group in row:
            series.extend(sub_group)
    return series


def create_sub_row_groups(orig_points: list[GcodePoint]) -> list[list[GcodePoint]]:
    def of_continuous_row_group(point_1: GcodePoint, point_2: GcodePoint) -> bool:
        """returns True if both points belong to the same face_group and are on edges that enclose the same triangle"""
        point_1_face_group = list(point_1.face_group_id) if isinstance(point_1.face_group_id, tuple) else [point_1.face_group_id]
        point_1_neighbors = [a for a in list(point_1.neighbor_tris) if not a is None]

        point_2_face_group = list(point_2.face_group_id) if isinstance(point_2.face_group_id, tuple) else [point_2.face_group_id]
        point_2_neighbors = [a for a in list(point_2.neighbor_tris) if not a is None]

        return (point_1_face_group[0] in point_2_face_group or point_1_face_group[-1] in point_2_face_group) and (point_1_neighbors[0] in point_2_neighbors or point_1_neighbors[-1] in point_2_neighbors)

    points = list(orig_points)
    all_groups = []
    def helper(starting_point) -> list[GcodePoint]:
        for i, check_point in enumerate(points):
            if of_continuous_row_group(starting_point, check_point):
                next_point = points.pop(i)
                return [next_point, *helper(next_point)]
        # this point is reached once all points have been checked and none are of the same sub row group as the starting point
        return []

    while points:
        next_point = points.pop(0)
        all_groups.append([next_point, *helper(next_point)])

    return all_groups


def arrange_gcode_points(gcode_points: list[GcodePoint]) -> dict[float, list[list[GcodePoint]]]:
    grouped_points = {} #{row y-value: list[list[GcodePoint]]}
    for point in gcode_points:
        y = point.coordinate[1]
        if y not in grouped_points:
            grouped_points[y] = [point]
        else:
            grouped_points[y].append(point)

    for row_y, row_points in grouped_points.items():
        sub_row_groups = create_sub_row_groups(row_points)
        assert len(row_points) == len_nested(sub_row_groups), f"Error! Point count mismatch: input: {len(row_points)}, output: {len_nested(sub_row_groups)}, y={row_y}"
        for j, sub_group in enumerate(sub_row_groups):
            #sub_group: list[GcodePoints]
            sub_row_groups[j].sort(key=sort_by_component) #sort sub_row_group coords by their x-value
        sub_row_groups.sort(key=sort_by_component_group) #sort sub_row_groups by the x-value of each group's first coordinate

        grouped_points[row_y] = sub_row_groups

    grouped_points = dict(sorted(grouped_points.items())) #sort point rows by y
    return grouped_points


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

        row_y_values = [i for i in np.arange(min_y + prefs["pass_spacing"] / 2, max_y, prefs["pass_spacing"])]
        print(f"{len(row_y_values)} slice planes generated")

        all_mesh_groups = []
        all_mesh_boundaries = []
        for group in object3d.face_groups:
            all_mesh_groups.append(object3d.parse_mesh(group))
            all_mesh_boundaries.append(object3d.parse_mesh(group, boundaries_only=True))
        print(f"{len(all_mesh_groups)} mesh groups generated")
        print(f"{len(all_mesh_boundaries)} mesh boundaries generated")
        face_groups = {a: [] for a in range(len(all_mesh_groups))} # {group id: [point rows belonging to group]}

        gcode_points = []
        for i, edge_id in enumerate(object3d.mesh):
            edge = (object3d.vertices[edge_id[0]], object3d.vertices[edge_id[1]])
            if edge[0][1] == edge[1][1] and edge[0][1] in row_y_values:
                print(f"COLLINEAR INTERSECTION FOUND")
                gcode_points.append(GcodePoint(np.array(edge[0]), edge_id, object3d))
                gcode_points.append(GcodePoint(np.array(edge[1]), edge_id, object3d))
                continue

            y_intersects = (np.array(row_y_values)[(max(edge[0][1],edge[1][1]) > np.array(row_y_values)) & (np.array(row_y_values) > min(edge[0][1],edge[1][1]))]).tolist()
            print(f"{i}, {edge}: {y_intersects}")
            if not y_intersects:
                continue
            y_intersects_idxs = range(row_y_values.index(y_intersects[0]), row_y_values.index(y_intersects[-1])+1)
            assert len(y_intersects) == len(y_intersects_idxs), f"Error! y_intersects and y_intersects_idxs do not length match."
            assert row_y_values[y_intersects_idxs[0]] == y_intersects[0], f"Error! y_intersect_idxs do not correlate correctly\nrow_y_values: {row_y_values[y_intersects_idxs[0]]}, y_intersects: {y_intersects[0]}"

            for y in y_intersects:
                A = np.array([[1, 0, edge[0][0] - edge[1][0]],
                              [0, 1, edge[0][2] - edge[1][2]],
                              [0, 0, edge[1][1] - edge[0][1]]])
                b = np.array([edge[0][0], edge[0][2], y - edge[0][1]])
                result = np.linalg.solve(A,b)
                coordinate = np.array([result[0],y,result[1]])
                gcode_points.append(GcodePoint(coordinate, edge_id, object3d))

        print(f"object {object3d.id}: {len(object3d.mesh)} edges processed, {len(gcode_points)} intersections found")
        gcode_point_rows = arrange_gcode_points(gcode_points)

        gcode_points = list(gcode_point_rows.values())

        #NEED TO ALTERNATE ROWS!!

        gcode_points = add_group_outline(gcode_points, all_mesh_boundaries, object3d)
        print(f"object {object3d.id}: added group outlines")


        #changed gcode_points to simple arrays

        gcode_points = untransform_face_groups(gcode_points, rotation_matrix)
        print(f"object {object3d.id}: untransformed point series")

        gcode_points = append_gcode_commands(gcode_points)
        print(f"object {object3d.id}: added gcode commands to coordinates")

        command_series = consolidate_groups(gcode_points)
        print(f"object {object3d.id}: consolidated all face groups into stream of commands")

        gcode = generate_gcode(command_series)
        print(f"object {object3d.id}: gcode generated")

        gcode_lines.extend(gcode)

    return gcode_lines
    print("writing to file")
    write_lines(sys.argv[2], gcode_lines)
    print(f"successfully wrote: {sys.argv[2]}")
