import numpy as np
import ssg_utils as utils


def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


def cal_glocal_position(object, floor, distance_rate=1.6):
    tgt_pos = object.position
    room_pos = floor.position
    room_rect = floor.bottom_rect

    # center
    center_dis = utils.euclideanDistance(tgt_pos, room_pos, 2)
    if center_dis < distance_rate:
        return 'in the center'

    # corner
    for point in room_rect:
        if utils.euclideanDistance(tgt_pos, point, 2) < distance_rate:
            return 'in the corner'

    return None


def cal_camera_relations(ObjNode_dict, camera_position, camera_view, inst_dict, floor_idx, fov = 60):
    relationships = []
    for obj_id in ObjNode_dict:
        if ObjNode_dict[obj_id].label == 'floor': continue

        # camera relation
        obj_position = ObjNode_dict[obj_id].position
        vector = obj_position - camera_position
        vector = vector / np.linalg.norm(vector)
        angle = utils.get_theta(vector, camera_view)

        a, b, c = getLinearEquation(camera_view[0]+camera_position[0],
                                    camera_view[1]+camera_position[1],
                                    camera_position[0],
                                    camera_position[1])

        if abs(angle) < fov/2:
            rela = 'in front of'
        elif abs(angle) > 180 - fov/2:
            rela = 'behind'
        elif a*obj_position[0] + b*obj_position[1] + c > 0:
            rela = 'right' if camera_view[1] > 0 else 'left'
        else:
            rela = 'left' if camera_view[1] > 0 else 'right'

        relationships.append(['-1', obj_id, rela])

        # global relation
        if inst_dict[ObjNode_dict[obj_id].label] > 1:
            rela = cal_glocal_position(ObjNode_dict[obj_id], ObjNode_dict[floor_idx])
            if rela is not None:

                # print(ObjNode_dict[obj_id].label, rela)
                # ObjNode_dict[obj_id].display_obb_box()
                relationships.append([obj_id, obj_id, rela])

    return relationships
