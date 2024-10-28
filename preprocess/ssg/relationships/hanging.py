import ssg_utils as utils


def cal_above_below_relationships(ObjNode_dict, src, scene_high):

    above_below_relationships  = []

    rect = src.bottom_rect
    src_max = src.z_max
    src_min = src.z_min
    src_pos = src.position

    for tgt_id in ObjNode_dict:
        tgt = ObjNode_dict[tgt_id]

        if tgt.label == 'floor' : continue

        tgt_max = tgt.z_max
        tgt_min = tgt.z_min
        tgt_pos = tgt.position
        tgt_rect = tgt.bottom_rect

        if utils.euclideanDistance (tgt.position, src.position, 2) < scene_high * 0.85: # make sure in same room
            # above
            if src_min > tgt_max and ( utils.if_inPoly(rect, tgt_pos) or utils.if_inPoly(tgt_rect, src_pos) ) :
                above_below_relationships.extend(utils.generate_relation(src.id, tgt_id, 'high'))

    return above_below_relationships


def filter_labels(obj_label):

    no_hanging_labels = ['floor', 'table', 'chair', 'desk', 'bottle']
    for l in no_hanging_labels:
        if l in obj_label:return False

    return True


def cal_hanging_relationships (ObjNode_dict, no_supported_objs, camera_angle,scene_high, dataset='scannet'):
    hanging_relationships = []

    for obj_id in ObjNode_dict:
        if obj_id not in no_supported_objs:
            obj = ObjNode_dict[obj_id]
            if not filter_labels(obj.label): continue
            desp = utils.generate_relation(obj.id, -2, 'hang')
            if 'tv' in obj.label:
                desp[2] = 'mounted on'
            if 'mirror' in obj.label:
                desp[2] = 'affixed to'

            hanging_relationships.append(desp)
            hanging_relationships.extend(cal_above_below_relationships(ObjNode_dict, obj, scene_high))

    return hanging_relationships
