import numpy as np
from numpy import ndarray as Array

prefs = {}

def read_lines(infile):
    with open(infile, "r") as file:
        lines = file.readlines()
        return lines

def write_lines(outfile, lines):
    with open(outfile, "w") as file:
        file.writelines(lines)


def get_config_data(infile):
        lines = read_lines(infile)
        user_prefs = {}
        for line in lines:
            key = line[:line.find("=")]
            value = line[line.find("=") + 1:line.find(";")]
            value = value.strip('"')
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    else:
                        value = str(value)

            user_prefs[key] = value

        user_prefs["nozzle_angle"] = user_prefs["nozzle_angle"] * (np.pi / 180)
        user_prefs["scan_angle"] = user_prefs["scan_angle"] * (np.pi / 180)
        user_prefs["face_tolerance"] = user_prefs["face_tolerance"] * (np.pi / 180)

        global prefs
        prefs = user_prefs
        return user_prefs


def create_rigid_transformation(translate: tuple|None, rotate: tuple|None, scale: tuple|None, R=3):
    if translate == 0 or translate is None:
        translate = (0,0,0)
    if rotate == 0 or rotate is None:
        rotate = (0,0,0)
    if scale == 0 or scale is None:
        scale = (1,1,1)

    assert len(translate) == len(rotate) == len(scale) == R, f"All transformations must be of length R! R={R}; Received {translate, rotate, scale}"

    tx, ty, tz = translate
    rx, ry, rz = rotate
    sx, sy, sz = scale

    T = np.array([[1,0,0,tx],
                  [0,1,0,ty],
                  [0,0,1,tz],
                  [0,0,0,1]])
    Rx = np.array([[1,0,0,0],
                   [0,np.cos(rx),-np.sin(rx),0],
                   [0,np.sin(rx),np.cos(rx),0],
                   [0,0,0,1]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry),0],
                   [0,1,0,0],
                   [-np.sin(ry),0,np.cos(ry),0],
                   [0,0,0,1]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0,0],
                   [np.sin(rz),np.cos(rz),0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    R = np.linalg.multi_dot((Rx,Ry,Rz))
    S = np.array([[sx,0,0,0],
                  [0,sy,0,0],
                  [0,0,sz,0],
                  [0,0,0,1]])
    return (T @ R @ S)


def rtrans(transform:Array, vector:Array, inverse=False):
    """Applies a 4x4 rigid transformation matrix to a given 1x3 array. Returns new vector."""
    if inverse:
        transform = np.linalg.inv(transform)
    return np.delete((transform @ np.append(vector, 1).T).T, 3)


def vector_angle(vector_1, vector_2):
    """
    returns the angle (in rad) between two vectors
    >>> vector_1 = np.array([5,-2,3])
    >>> vector_2 = np.array([-4,5,7])
    >>> round(float(vector_angle(vector_1, vector_2)), 4)
    1.7253
    """
    return np.atan2(np.linalg.norm(np.cross(vector_1, vector_2)), np.dot(vector_1, vector_2))





class Object3D:
    def __init__(self, id=0, component_id=0, ob_type="", vertices: list[Array] = [], name="", tris: list = [],
                 transform: Array = np.identity(4), printable=True):
        self.id = id
        self.component_id = component_id
        self.ob_type = ob_type
        self.vertices = vertices # list[array[float, float, float]]
        self.name = name  #
        self.tris = tris  #
        self.transform = transform
        self.printable = printable
        self.tri_edges = {} #{edge:tuple(int,int) : (Triangle3D, Triangle3D)}
        self.mesh = []      #[edge:tuple(int,int)]
        self.face_groups = []   #[[Triangle3D]]


    def __str__(self):
        return f"<Object3D-{self.id}-{self.name}>"


    def apply_transform(self, inverse=False, matrix=np.identity(4)):
        """Applies a rigid transformation to the object. If no 4x4 transformation matrix is provided, the object's self.transform transformation will be applied."""
        temp_transform = self.transform
        if not np.array_equal(matrix, np.identity(4)):
            if matrix.shape != (4, 4):
                raise ValueError("Object3D: Transformation matrix must be 4x4!")

            temp_transform = matrix

        if inverse:
            temp_transform = np.linalg.inv(temp_transform)

        for i in range(len(self.vertices)):
            self.vertices[i] = rtrans(temp_transform, self.vertices[i])
        print(f"object {self.id} apply_transform(): applied transform to {len(self.vertices)}")


    def parse_mesh(self, tri_list=[], boundaries_only=False):
        """
        writes a list [(int, int)] of all individual edges to object.mesh\n
        if tri_list[Triangle3D] is provided, function returns list of all individual edges from tri_list
        boundaries_only=True requires a tri_list argument
        """
        assert boundaries_only and tri_list or not boundaries_only, "Error! Object3D.parse_mesh(boundaries_only=True) requires a tri_list argument!"
        all_edges = []
        tri_group = tri_list
        if not tri_list:
            tri_group = self.tris
        for tri in tri_group:
            for edge, neighbor in zip(tri.edges(True), tri.neighbors()):
                if edge not in all_edges:
                    if not boundaries_only:
                        all_edges.append(edge)
                    elif boundaries_only and (None in self.tri_edges[edge] or neighbor not in tri_group):
                        all_edges.append(edge)
                    else:
                        pass
        if not tri_list:
            self.mesh = all_edges
        else:
            return all_edges
        print(f"object {self.id} parse_mesh(): {len(all_edges)}/{len(self.tris)*3} edges added to mesh")


    def parse_tri_edges(self):
        """writes a dictionary {(int, int): (Triangle3D, Triangle3D)} of all edges and the tris that use them to object.tri_edges"""
        edge_tris = {}
        for tri in self.tris:
            for edge in tri.edges(True):
                if edge not in edge_tris:
                    edge_tris[edge] = (tri, None)
                else:
                    edge_tris[edge] = (edge_tris[edge][0], tri)

        self.tri_edges = edge_tris
        print(f"object {self.id} parse_tri_edges(): {len(edge_tris)} edges indexed")


    def fg_boundary_edges(self, use_face_groups=True) -> list[tuple[int,int]] | list[list[tuple[int,int]]]:
        """
        returns a list of all tri_edges.keys() whose values() have one None
        if face_groups=True, returns multiple lists of boundary tri_edges, one list for every face_group
        *ensure tri_edges is up to date before running
        """
        all_boundary_edges = []
        if not self.tri_edges:
            self.parse_tri_edges()

        if use_face_groups and self.face_groups:
            for face in self.face_groups:
                face_boundary_edges= []
                for tri in face:
                    for edge in tri.edges(True):
                        if None in self.tri_edges[edge] and not edge in face_boundary_edges:
                            face_boundary_edges.append(edge)

                all_boundary_edges.append(face_boundary_edges)

            return all_boundary_edges

        elif use_face_groups and not self.face_groups:
            print(f"Error! Object {self.id} face_groups was empty")

        return []


class Triangle3D:
    def __init__(self, parent_object: Object3D, vertex_ids: tuple, support=""):
        self.parent = parent_object
        self.v_ids = vertex_ids
        self.support = support
        self.face_group_id = -1

    def v(self, vertex: int):
        assert 0 < vertex < 4, "Error! Triangle3D.v() requires a vertex integer between 1 and 3 inclusive!"
        return self.parent.vertices[self.v_ids[vertex - 1]]

    # def __str__(self):
    #     return f"<Triangle3D-{self.v(1)}-{self.v(2)}-{self.v(3)}>"

    def normal(self):
        """Returns the unit normal of the triangle in R3"""
        cross = np.cross(self.v(2) - self.v(1), self.v(3) - self.v(1))
        norm = np.linalg.norm(cross)
        return cross / norm

    def vertices(self, vertex_ids=False) -> tuple[int, int, int] | tuple[Array,Array,Array]:
        if vertex_ids:
            return self.v_ids
        else:
            return self.v(1), self.v(2), self.v(3)

    def edges(self, vertex_ids=False) -> tuple | Array:
        """
        If vertex_ids is set to True, method will return the vertex id of each edge instead of its coordinate in 3D space.
        Edge vertex_ids are sorted from least to greatest.Eedge vertices are not sorted
        """
        if vertex_ids:
            e1, e2, e3 = (self.v_ids[1], self.v_ids[0]), (self.v_ids[2], self.v_ids[1]), (self.v_ids[0], self.v_ids[2])
            e1 = tuple(sorted(e1))
            e2 = tuple(sorted(e2))
            e3 = tuple(sorted(e3))
            return e1, e2, e3
        else:
            return (self.v(2), self.v(1)), (self.v(3), self.v(2)), (self.v(1), self.v(3))

    def neighbors(self):
        neighbors = []
        for edge in self.edges(True):
            tri_1, tri_2 = self.parent.tri_edges[edge]
            neighbors.extend([tri_1, tri_2])
        neighbors = [i for i in neighbors if i != self]
        return neighbors


class GcodePoint:
    def __init__(self, coordinate:Array, parent_edge:tuple[int,int], parent_object3d:Object3D, face_group_id:int=-1):
        self.coordinate = coordinate
        self.parent_edge = parent_edge
        self.parent_object3d = parent_object3d
        self.neighbor_tris = self.parent_object3d.tri_edges[self.parent_edge]
        self.face_group_id = self.get_face_group_id() if face_group_id == -1 else face_group_id
        # if a face_group_id argument is provided, that will be the attribute. Else, the face_group_id will be found automatically.

    def get_face_group_id(self) -> int | tuple[int, int]:
        """Returns a GcodePoint's face_group id. If point lies on border between two face_groups, both groups' ids will be returned."""
        fg_id_1 = self.neighbor_tris[0].face_group_id
        fg_id_2 = self.neighbor_tris[1].face_group_id
        if fg_id_1 == fg_id_2:
            return fg_id_1
        elif fg_id_1 is None:
            return fg_id_2
        elif fg_id_2 is None:
            return fg_id_1
        elif fg_id_1 != fg_id_2:
            return fg_id_1, fg_id_2
        else:
            raise ValueError("auxiliary.GcodePoint.get_face_group_id(): Erroneous neighboring tri face group data")

    def face_group(self) -> list[Triangle3D] | tuple[list[Triangle3D], list[Triangle3D]]:
        """Returns the GcodePoint's face_group tri list. If point belongs to two face_groups, both lists will be returned."""
        if isinstance(self.face_group_id, int):
            return self.parent_object3d.face_groups[self.face_group_id]
        elif isinstance(self.face_group_id, tuple) and len(self.face_group_id) == 2:
            return self.parent_object3d.face_groups[self.face_group_id[0]], self.parent_object3d.face_groups[self.face_group_id[1]]
        else:
            raise ValueError(f"auxiliary.GcodePoint.face_group(): Invalid self.face_group_id value: {self.face_group_id}")

    def is_border(self) -> bool:
        """returns True if GcodePoint lies on an edge that touches only one tri or is shared by tris of different face_groups."""
        if None in self.neighbor_tris:
            # Point lies on edge that encloses only one triangle
            return True
        elif self.neighbor_tris[0].face_group_id != self.neighbor_tris[1].face_group_id:
            # Point lies on edge that touches two triangles that are not of the same face group
            return True
        else:
            return False


if __name__ == "__main__":
    print(np.array([1, "", 3]))
