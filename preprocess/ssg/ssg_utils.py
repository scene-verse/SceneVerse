import trimesh
import math
from shapely import geometry
import os
import numpy as np
import pyvista as pv
from ssg_data.dictionary import *
import random
import open3d as o3d


def cw_rotate(point, ang):
    x,y,_ = point
    ang = math.radians(ang)
    new_x = round(x * math.cos(ang) - y * math.sin(ang), 5)
    new_y = round(x * math.sin(ang) + y * math.cos(ang), 5)
    return new_x, new_y

def euclideanDistance(instance1, instance2, dimension):
    distance = 0
    for i in range(dimension):
        distance += (instance1[i] - instance2[i])**2

    return math.sqrt(distance)

def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

def get_Poly_Area(polygon):

    line = geometry.LineString(polygon)
    polygon = geometry.Polygon(line)
    return polygon.area

def get_theta (x, y):

    x = np.array(x)
    y = np.array(y)

    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))

    dian = x.dot(y)

    cos_ = dian / (l_x * l_y)

    angle_hu = np.arccos(cos_)
    angle_d = angle_hu * 180 / np.pi

    return angle_d

def generate_relation(src, tgt, express):
    if 'oppo_support' in express:
        oppo_rels = [tgt, src, random.choice(opp_support_express)]
        return oppo_rels
    elif 'support' in express:
        rels = [src, tgt, random.choice(support_express)]
        return rels
    elif 'embed' in express:
        oppo_rels = [tgt, src, random.choice(opp_embed_express)]
        return oppo_rels
    elif 'inside' in express:
        oppo_rels = [tgt, src, random.choice(opp_inside_express)]
        return oppo_rels
    elif 'hang' in express:
        oppo_rels = [src, tgt, random.choice(hanging_express)]
        return oppo_rels
    elif 'under' in express:
        oppo_rels = [src, tgt, random.choice(under_express)]
        return oppo_rels
    elif 'close' in express:
        oppo_rels = [src, tgt, random.choice(close_express)]
        return oppo_rels
    elif 'high' in express:
        rels = [src, tgt, random.choice(above_express)]
        oppo_rels = [tgt, src, random.choice(below_express)]
        return [rels,oppo_rels]

def visualize_relations(target_obj, obj, relationship, camera_angle, camera_position = np.array([0,0,0]), save = False):
    if save:
        render_bbox_pyvista(obj, target_obj, relationship, camera_angle, camera_position)
    else:
        axis_align_matrix = target_obj.align_matrix

        tgt_mesh = trimesh.load(target_obj.obj_mesh)
        src_mesh = trimesh.load(obj.obj_mesh)
        tgt_mesh.apply_transform(axis_align_matrix)
        src_mesh.apply_transform(axis_align_matrix)

        tgt_p = tgt_mesh.bounding_box.as_outline()
        tgt_p.entities[0].color = (255, 0, 0, 255)

        src_p = src_mesh.bounding_box.as_outline()
        src_p.entities[0].color = (255, 255, 0, 255)

        scene_mesh = trimesh.load_mesh(target_obj.scan_mesh)

        scene_mesh.apply_transform(axis_align_matrix)


        # draw line of two objects
        lines_of_center = [[np.array(target_obj.position), np.array(obj.position)]]
        p = trimesh.load_path(lines_of_center)

        # rotate from camera view
        camera_rotate = trimesh.transformations.rotation_matrix(
            np.deg2rad(camera_angle), [0,0,1], point=(0,0,0)
        )

        scene_mesh.apply_transform(camera_rotate)
        tgt_p.apply_transform(camera_rotate)
        src_p.apply_transform(camera_rotate)
        p.apply_transform(camera_rotate)

        # draw camera center
        camera = trimesh.primitives.Sphere(radius=0.2, center=camera_position)
        camera.apply_transform(camera_rotate)

        Scene = trimesh.Scene()

        camera_rotate = trimesh.transformations.rotation_matrix(
           -20, [1,0,0], point=(0,0,0)
        )
        Scene.add_geometry([scene_mesh, src_p, tgt_p, p])
        Scene.apply_transform(camera_rotate)

        Scene.show()

def visualize_relations_multi_objs(objs, relationship, item, camera_angle, camera_position = np.array([0,0,0]), save = False):
    # img save name
    save_img_name = '_'.join([relationship, objs[0].label]) + str(item)

    # load mesh
    scene_mesh = pv.read(objs[0].scan_ply)
    axis_align_matrix = objs[0].align_matrix
    tgt_meshs = [trimesh.load(obj.obj_mesh) for obj in objs]

    # show results
    plotter = pv.Plotter(off_screen=True)
    light = pv.Light(light_type='headlight', intensity=0.3)
    plotter.add_light(light)

    # draw camera
    camera_look_at = cw_rotate(camera_position+np.array([0,1,0]), -camera_angle)
    camera_look_at = np.array([camera_look_at[0], camera_look_at[1], 0])
    # plotter.add_lines(np.array([camera_position, camera_look_at]), color='blue', width=3)
    mesh = pv.Arrow(start=camera_position, direction=camera_look_at)
    plotter.add_mesh(mesh)

    # added scene mesh
    plotter.add_mesh(scene_mesh.transform(axis_align_matrix), rgb=True)

    # rotate to axis align and added in to scene
    for tgt_mesh in tgt_meshs:
        tgt_mesh.apply_transform(axis_align_matrix)
        # draw bbox
        tgt_points = np.array(tgt_mesh.bounding_box.as_outline().vertices)
        tgt_edges = np.array(tgt_mesh.bounding_box.as_outline().vertex_nodes)
        tgt_points_new = []
        for edge in tgt_edges:
            tgt_points_new.append(tgt_points[edge[0]])
            tgt_points_new.append(tgt_points[edge[1]])

        plotter.add_lines(np.array(tgt_points_new), color='yellow', width=3)

    plotter.add_point_labels(
        [np.array(obj.position) for obj in objs],
        [obj.label for obj in objs],
        margin=0,
        fill_shape=True,
        font_size=18,
        shape_color="black",
        point_color="red",
        text_color="white",
        always_visible=True,
    )

    plotter.add_text(
        save_img_name,
        position='upper_right',
        color='Blue',
        shadow=True,
        font_size=19,
    )

    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 90 - camera_angle + 180
    plotter.camera.elevation = 65

    plotter.camera.zoom(1.2)
    plotter.show()

def render_bbox_pyvista(tgt, src, relationship, camera_angle, camera_position):

    # img save name
    save_img_name = '_'.join([relationship, src.label, src.id, tgt.label, tgt.id])


    # load mesh
    tgt_mesh = trimesh.load(tgt.obj_mesh)
    src_mesh = trimesh.load(src.obj_mesh)
    scene_mesh = pv.read(tgt.scan_ply)
    axis_align_matrix = tgt.align_matrix

    # rotate to axis align
    tgt_mesh.apply_transform(axis_align_matrix)
    src_mesh.apply_transform(axis_align_matrix)

    # draw bbox
    tgt_points = np.array(tgt_mesh.bounding_box.as_outline().vertices)
    tgt_edges = np.array(tgt_mesh.bounding_box.as_outline().vertex_nodes)
    tgt_points_new = []
    for edge in tgt_edges:
        tgt_points_new.append(tgt_points[edge[0]])
        tgt_points_new.append(tgt_points[edge[1]])

    src_points = np.array(src_mesh.bounding_box.as_outline().vertices)
    src_edges = np.array(src_mesh.bounding_box.as_outline().vertex_nodes)
    src_points_new = []
    for edge in src_edges:
        src_points_new.append(src_points[edge[0]])
        src_points_new.append(src_points[edge[1]])

    # show results
    plotter = pv.Plotter(off_screen=True)
    light = pv.Light(light_type='headlight', intensity=0.3)
    plotter.add_light(light)

    # draw camera
    camera_look_at = cw_rotate(camera_position+np.array([0,1,0]), -camera_angle)
    camera_look_at = np.array([camera_look_at[0], camera_look_at[1], 0])
    # plotter.add_lines(np.array([camera_position, camera_look_at]), color='blue', width=3)
    mesh = pv.Arrow(start=camera_position, direction=camera_look_at)
    plotter.add_mesh(mesh)

    plotter.add_mesh(scene_mesh.transform(axis_align_matrix), rgb=True)
    plotter.add_lines(np.array([src.position, tgt.position]), color='red', width=3)
    plotter.add_lines(np.array(src_points_new), color='red', width=3)
    plotter.add_lines(np.array(tgt_points_new), color='yellow', width=3)
    # plotter.add_axes_at_origin()

    plotter.add_point_labels(
        [
            src.position,
            tgt.position,
            camera_position
        ],
        [src.label, tgt.label, 'Camera View'],
        margin=0,
        fill_shape=True,
        font_size=18,
        shape_color="black",
        point_color="red",
        text_color="white",
        always_visible=True,
    )

    plotter.add_text(
        save_img_name,
        position='upper_right',
        color='Blue',
        shadow=True,
        font_size=19,
    )

    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 90 - camera_angle + 180
    plotter.camera.elevation = 65

    plotter.camera.zoom(1.2)
    plotter.show()

def visualize_camera_relations(ObjNode_dict, camera_relations, camera_position, camera_view, save = False):
    tgt = ObjNode_dict[camera_relations[0][1]]
    scene_mesh = trimesh.load(tgt.scan_mesh)
    axis_align_matrix = tgt.align_matrix
    objs_mesh = []
    for rela in camera_relations:
        _, obj, desc = rela
        obj = ObjNode_dict[obj]
        src_mesh = trimesh.load(obj.obj_mesh)
        src_mesh.apply_transform(axis_align_matrix)
        src_p = src_mesh.bounding_box.as_outline()

        if desc == 'behind':
            src_p.entities[0].color = (0, 255, 0, 255)
        elif desc == 'in front of':
            src_p.entities[0].color = (255, 0, 0, 255)
        elif desc == 'left':
            src_p.entities[0].color = (0, 0, 255, 255)
        else:
            src_p.entities[0].color = (0, 255, 255, 255)

        objs_mesh.append (src_p)

    end_point = np.array(camera_position) + np.array(camera_view)
    # draw line of two objects
    lines_of_center = [[end_point, np.array(camera_position)],]
    p = trimesh.load_path(lines_of_center)
    scene_mesh.apply_transform(axis_align_matrix)

    # camera position
    camera_pos = trimesh.primitives.Sphere(radius=0.2, center=np.array(camera_position))

    Scene = trimesh.Scene()
    Scene.add_geometry([scene_mesh, p, camera_pos])
    Scene.add_geometry(objs_mesh)

    if not save:
        Scene.show()
    else:
        data = Scene.save_image(resolution=(640, 640))
        save_img_name = tgt.scan_id + 'camera_view.png'
        save_path = os.path.join('../SSGResults/cameras', save_img_name)
        with open(save_path, 'wb') as f:
            f.write(data)
        #Scene.show()


def read_one_obj(bbox_points, scene_file):
    scene_mesh = pv.read(scene_file)
    scene_points = scene_mesh.points

    # visualize scene
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(scene_points)


    bbox_center = np.mean(bbox_points, axis=0)
    bbox_size = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
    gt_o3d_box = o3d.geometry.OrientedBoundingBox(bbox_center, np.eye(3, 3), bbox_size)
    gt_o3d_box.color = [0, 1, 0]

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-0, -0, -0])
    o3d.visualization.draw_geometries([o3d_pcd, gt_o3d_box, mesh_frame])
