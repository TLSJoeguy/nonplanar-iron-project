import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from numpy import ndarray as Array
from auxiliary import Object3D, Triangle3D


def print_added_object(object_added: Object3D):
    """prints various data of an object"""
    print(f"Added obj {object_added.id}: type={object_added.ob_type}, name={object_added.name}, tri count={len(object_added.tris)}, printable={object_added.printable}")


def parse_transform_data(data: str) -> Array:
    """takes an item's transform data and reformats it to a 4x4 rigid transformation matrix"""
    matrix = np.array([float(i) for i in data.split(" ")])
    matrix = matrix.reshape(4,3)
    matrix = np.column_stack((matrix, [0, 0, 0, 0]))
    matrix[:, 3] = matrix[3].tolist()
    matrix[3] = [0.0, 0.0, 0.0, 1.0]
    return matrix


def parse_tri_data(parent_object: Object3D, triangle_elements) -> list[Triangle3D]:
    """takes in two lists of XML elements (one containing vertex child elements, one containing triangle child elements) and returns Triangle3D objects"""
    assert parent_object.vertices,f"Parent object {parent_object} does not contain vertices!"
    triangles = []
    object_tris = []
    edge_dict = {}

    for triangle_element in triangle_elements:
        triangles.append((int(triangle_element.get("v1")), int(triangle_element.get("v2")), int(triangle_element.get("v3")), triangle_element.get("{http://schemas.slic3r.org/3mf/2017/06}custom_supports", None)))

    for data in triangles:
        support_data = data[3]
        vids = data[:-1]

        new_triangle = Triangle3D(parent_object, vids, support_data)
        object_tris.append(new_triangle)

    return object_tris


def parse_vertices(vertex_elements) -> list:
    """Receives the resulting iterable of an XML findall query and returns a list of 1x3 arrays"""
    vertices = []
    for vertex_element in vertex_elements:
        vertices.append(np.array([float(vertex_element.get("x")),
                                  float(vertex_element.get("y")),
                                  float(vertex_element.get("z"))]))
    return vertices


def parse_tri_data_vertex(object3d: Object3D):
    vertices_shared = {}
    for tri in object3d.tris:
        for v in tri.v_ids:
            if v not in vertices_shared.keys():
                vertices_shared[v] = []
            vertices_shared[v] = vertices_shared[v].append(tri)

    return vertices_shared


def parse_3mf(input_file: str) -> dict:
    """receives a .3MF file reference and returns the model data formatted for usability"""
    pref = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
    objects = {}

    with (zipfile.ZipFile(input_file, "r") as archive_3mf):
        with archive_3mf.open("3D/3dmodel.model") as model_file, archive_3mf.open("Metadata/Slic3r_PE_model.config") as config_file:
            root_model = ET.parse(model_file).getroot()
            root_config = ET.parse(config_file).getroot()

            for object_element, build_item_element in zip(root_model.iter(pref+"object"),root_model.iter(pref+"item")):
                current_object_id = int(object_element.get("id",0))

                new_object = Object3D()
                new_object.id = current_object_id
                new_object.ob_type = object_element.get("type")
                new_object.printable = bool(build_item_element.get("printable"))
                new_object.transform = parse_transform_data(build_item_element.get("transform",""))

                if object_element.find(pref+"components") is None:      #if the object is not a clone of another
                    new_object.name = root_config.findall(f".//object[@id='{str(current_object_id)}']")[0][0].get("value")   #XPath search: all <object> elements with attribute "id" of value (current_object_id)
                    new_object.vertices = parse_vertices(object_element.iter(pref+"vertex"))
                    new_object.tris = parse_tri_data(new_object, object_element.iter(pref+"triangle"))

                else:
                    parent_id = int(object_element[0][0].get("objectid", 0))  # object>components>component.get("objectid")
                    parent_object = objects[parent_id]

                    new_object.component_id = parent_object.id
                    new_object.name = parent_object.name
                    new_object.vertices = parent_object.vertices
                    new_object.tris = parent_object.tris

                # Add new_object to objects dictionary
                objects[current_object_id] = new_object
                print_added_object(new_object)

        archive_3mf.close()

    return objects