import numpy as np
import networkx as nx
import itertools

import ssg_utils as utils


def are_furniture_aligned(furniture1, furniture2, offset_threshold):
    x1, y1, z1 = furniture1['center']
    x2, y2, z2 = furniture2['center']
    h1 = furniture1['size']
    h2 = furniture2['size']
    rect1 = furniture1['rect']
    rect2 = furniture2['rect']

    # x_offset
    x_offset = abs(x1 - x2)
    # y_offset
    y_offset = abs(y1 - y2)
    # z_offset
    z_offset = abs(z1 - z2)

    # volumn
    volumn_diff = abs(utils.get_Poly_Area(rect1) - utils.get_Poly_Area(rect2))

    if volumn_diff > offset_threshold:
        return False
    if z_offset > offset_threshold:
        return False

    if x_offset > offset_threshold and y_offset > offset_threshold:
        return False

    if x_offset < offset_threshold:
        return 'x'

    if y_offset < offset_threshold:
        return 'y'


def find_aligned_furniture(furniture_list, ObjNode_dict, offset_threshold):
    aligned_furniture = []

    for i, object_id1 in enumerate(furniture_list):
        obj1 = ObjNode_dict[object_id1]
        furniture1 = {'center': np.array(obj1.position), 'size': obj1.z_max - obj1.z_min, 'rect': obj1.bottom_rect}

        for j, object_id2 in enumerate(furniture_list[i+1:]):
            obj2 = ObjNode_dict[object_id2]
            furniture2 = {'center': np.array(obj2.position), 'size': obj2.z_max - obj2.z_min, 'rect': obj2.bottom_rect}
            is_aligned = are_furniture_aligned(furniture1, furniture2, offset_threshold)
            if is_aligned:
                aligned_group = [obj1.id, obj2.id, is_aligned]
                aligned_furniture.append(aligned_group)

    aligned_furniture_merge = furniture_merge_lists(aligned_furniture)
    return aligned_furniture_merge

def furniture_merge_lists(lists):
    merged_lists = []

    x_list = [lst[:2] for lst in lists if 'x' in lst]
    y_list = [lst[:2] for lst in lists if 'y' in lst]

    merged_x_list = merge_sublists(x_list)
    merged_y_list = merge_sublists(y_list)

    merged_lists.extend(merged_x_list)
    merged_lists.extend(merged_y_list)

    return merged_lists


def merge_sublists(L):
    length = len(L)
    for i in range(1, length):
        for j in range(i):
            if 0 in L[i] or 0 in L[j]:
                continue
            x = set(L[i]).union(set(L[j]))
            y = len(L[i]) + len(L[j])
            if len(x) < y:
                L[i] = list(x)
                L[j] = [0]

    return [i for i in L if 0 not in i]


def find_middle_furniture (proximity_relations, ObjNode_dict):
    # in the middle of
    middle_relationships = []
    G = nx.DiGraph()
    for (src, tgt, rel) in proximity_relations:
        G.add_edge(src, tgt, label=rel)

    edage_dict = G.edges.data()._adjdict
    for src_id in ObjNode_dict:
        if src_id not in edage_dict: continue
        if ObjNode_dict[src_id].label == 'floor' :continue
        neighbors = edage_dict[src_id]
        tgt_ids = list(neighbors.keys())
        combinations = list(itertools.combinations(tgt_ids, 2))

        for group in combinations:
            idx1, idx2 = group
            if 'near' in neighbors[idx1]['label'] and 'near' in neighbors[idx2]['label']:

                direction1 = int(neighbors[idx1]['label'].split(' ')[0])
                direction2 = int(neighbors[idx2]['label'].split(' ')[0])
                if abs(direction1 - direction2) == 6:
                    middle_relationships.append([[src_id,idx1,idx2], 'in the middle of'])

    return middle_relationships


if __name__ == '__main__':
    # UnitTest
    lists = [['26', '36', 'x'], ['26', '30', 'x'], ['29', '28', 'y'], ['29', '30', 'y'],
             ['28', '30', 'y'], ['28', '33', 'x'], ['35', '36', 'y'], ['35', '32', 'y'],
             ['35', '33', 'y'], ['31', '37', 'x'], ['2', '4', 'y'], ['2', '3', 'y'],
             ['34', '32', 'y'], ['34', '33', 'y'], ['37', '3', 'x'], ['36', '30', 'x'],
             ['4', '3', 'y'], ['32', '33', 'y']]
    output = furniture_merge_lists(lists)
    print(output)
