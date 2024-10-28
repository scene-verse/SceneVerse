import numpy as np
import itertools
import ssg_utils as utils

def get_direction(src_obj, tgt_obj):

    sx, sy = src_obj
    tx, ty = tgt_obj

    y = np.array((tx - sx, ty - sy))
    y = y / np.linalg.norm(y)

    angle_d = utils.get_theta(y, [1, 0])

    direction = round(angle_d / 30)


    if ty > sy : # tgt is up
        if direction == 0: return "3"
        elif direction == 1: return "2"
        elif direction == 2: return "1"
        elif direction == 3: return "12"
        elif direction == 4: return "11"
        elif direction == 5: return "10"
        elif direction == 6: return "9"
    else:
        if direction == 0: return "3"
        elif direction == 1: return "4"
        elif direction == 2: return "5"
        elif direction == 3: return "6"
        elif direction == 4: return "7"
        elif direction == 5: return "8"
        elif direction == 6: return "9"

def get_oppo_direction(direction):

    if direction in ['2', '3', '4']:
        return 'to the left of'
    elif direction in ['8', '9', '10']:
        return 'to the right of'
    elif direction in ['11','12','1']:
        return 'behind'
    else:
        return 'in front of'

def get_space_relations(src, tgt):
    overlap_point = 0
    tgt_rect = tgt.bottom_rect
    for point in tgt_rect:
        if utils.if_inPoly(src.bottom_rect, point): # have overlap
            overlap_point += 1

    return overlap_point

def get_distance(src, tgt):

    dis_of_center = utils.euclideanDistance(src.position[:2], tgt.position[:2], 2)
    src_w = utils.euclideanDistance(src.position[:2], src.bottom_rect[0][:2], 2)
    tgt_w = utils.euclideanDistance(tgt.position[:2], tgt.bottom_rect[0][:2], 2)

    return dis_of_center > 1.5 * (src_w + tgt_w)

def cal_proximity_relationships(neighbor_objs_id, camera_angle, ObjNode_dict, scene_high):
    proximity_relations = []

    relations = ''

    neighbor_objs_id_list = [i for i in range(len(neighbor_objs_id))]
    combinations = list(itertools.combinations(neighbor_objs_id_list, 2))

    for combination in combinations:

        src_idx, tgt_idx = combination
        src = neighbor_objs_id[src_idx]
        tgt = neighbor_objs_id[tgt_idx]

        if ObjNode_dict[src].room_id != ObjNode_dict[tgt].room_id:
            continue

        # is overlap
        overlap_points = get_space_relations(src=ObjNode_dict[src], tgt=ObjNode_dict[tgt])

        if overlap_points > 0 :
            # bulid in
            if overlap_points >=3:
                relations = 'under'
            # close to
            else:
                relations = 'close to'
            proximity_relations.append(utils.generate_relation(ObjNode_dict[src].id, ObjNode_dict[tgt].id, relations))
            proximity_relations.append(utils.generate_relation(ObjNode_dict[tgt].id, ObjNode_dict[src].id, relations))

        else:
            # direction
            src_obj_center = ObjNode_dict[src].position
            tgt_obj_center = ObjNode_dict[tgt].position

            src_obj_center_new = utils.cw_rotate(src_obj_center, camera_angle)
            tgt_obj_center_new = utils.cw_rotate(tgt_obj_center, camera_angle)

            if src_obj_center_new == tgt_obj_center_new:
                print ('src_obj_center_new == tgt_obj_center_new ', ObjNode_dict[src].id , ObjNode_dict[tgt].id)
                break
            direction = get_direction(src_obj_center_new, tgt_obj_center_new)

            oppo_direction = get_oppo_direction(direction)
            if get_distance(src=ObjNode_dict[src], tgt=ObjNode_dict[tgt]):
                relations = direction + ' o‘clock direction far from'

            else:
                relations = direction + ' o‘clock direction near'
            proximity_relations.append([ObjNode_dict[tgt].id, ObjNode_dict[src].id, relations])
            if oppo_direction is not None:
                proximity_relations.append([ObjNode_dict[src].id, ObjNode_dict[tgt].id, oppo_direction])

    return proximity_relations
