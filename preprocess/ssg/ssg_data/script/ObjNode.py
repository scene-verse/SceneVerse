import networkx as nx
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


class ObjNode(object):
    def __init__(self, id=None, label=None, mesh=None, position=None, size=None, children=[], room_id = None,dataset='scannet'):
        self.id = id
        self.label = label
        self.obj_mesh = mesh
        self.size = size
        self.position = position
        self.children = children
        self.room_id = room_id

        self.align_matrix, self.position, self.z_min, self.z_max, self.bottom_rect, self.top_rect = self.get_object_information(dataset)

    def __str__(self):
        return "[{}:{},{},{}]".format(self.id, self.label, self.position, self.angle)

    def get_object_information(self, dataset):
        position = self.position # - bias
        axis_align_matrix = None
        x_min = position[0] - self.size[0] / 2
        x_max = position[0] + self.size[0] / 2
        y_min = position[1] - self.size[1] / 2
        y_max = position[1] + self.size[1] / 2
        z_min = position[2] - self.size[2] / 2
        z_max = position[2] + self.size[2] / 2
        top_vertics = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min],
                                [x_max, y_max, z_min], [x_min, y_max, z_min]])
        bottom_vertics = np.array([[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max],
                               [x_min, y_max, z_max]])

        return axis_align_matrix, position, z_min, z_max, bottom_vertics, top_vertics

    def display_obb_box(self, scene_visible = True):

        axis_align_matrix = self.align_matrix

        obj_mesh = trimesh.load(self.obj_mesh)
        scene_ply = pv.read(self.scan_ply)

        # rotate to axis align
        obj_mesh.apply_transform(axis_align_matrix)

        if self.label == 'floor':

            scene_mesh = trimesh.load(self.scan_mesh)
            scene_mesh.apply_transform(axis_align_matrix)

            # draw aabb
            tgt_points = np.array(scene_mesh.bounding_box.as_outline().vertices)
            tgt_edges = np.array(scene_mesh.bounding_box.as_outline().vertex_nodes)
            tgt_points_new = []
            for edge in tgt_edges:
                tgt_points_new.append(tgt_points[edge[0]])
                tgt_points_new.append(tgt_points[edge[1]])

            # show results
            plotter = pv.Plotter(off_screen=False)
            light = pv.Light(light_type='headlight', intensity=0.2)
            plotter.add_light(light)

            plotter.add_mesh(scene_ply.transform(axis_align_matrix), rgb=True)
            plotter.add_lines(np.array(tgt_points_new), color='red', width=3)

        else:
            # draw bbox
            tgt_points = np.array(obj_mesh.bounding_box_oriented.as_outline().vertices)
            tgt_edges = np.array(obj_mesh.bounding_box_oriented.as_outline().vertex_nodes)
            tgt_points_new = []
            for edge in tgt_edges:
                tgt_points_new.append(tgt_points[edge[0]])
                tgt_points_new.append(tgt_points[edge[1]])

            # draw aabb
            aa_tgt_points = np.array(obj_mesh.bounding_box.as_outline().vertices)
            aa_tgt_edges = np.array(obj_mesh.bounding_box.as_outline().vertex_nodes)
            aa_tgt_points_new = []
            for edge in aa_tgt_edges:
                aa_tgt_points_new.append(aa_tgt_points[edge[0]])
                aa_tgt_points_new.append(aa_tgt_points[edge[1]])

            # show results
            plotter = pv.Plotter(off_screen=False)
            light = pv.Light(light_type='headlight', intensity=0.2)
            plotter.add_light(light)

            plotter.add_mesh(scene_ply.transform(axis_align_matrix), rgb=True)
            plotter.add_lines(np.array(tgt_points_new), color='red', width=3)
            plotter.add_lines(np.array(aa_tgt_points_new), color='yellow', width=3)

        plotter.camera.zoom(1.2)
        plotter.show()


if __name__ == '__main__':
    obj_sample = ObjNode(id=1, label='', size='',
                         mesh='../../DataAnnotation/data/scannet_objs/scene0000_00/45/mesh.obj')
    G = nx.DiGraph()
    G.add_node(obj_sample.id, desc = 'here1')
    G.add_node(obj_sample.id+1, desc = 'here2')
    G.add_node(obj_sample.id +3, desc = 'here3')
    G.add_edge(1, 2, name ='support')
    G.add_edge(2, 1, name='support2')
    pos = nx.spring_layout(G)
    nx.draw(G,pos)
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()
