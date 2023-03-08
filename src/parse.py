import json
import numpy as np

from utils.graphs import *

def get_from_json(fname, field_name):
    f_master = open(fname)
    data = json.load(f_master)
    return data[field_name] 

def get_update_time_split_thershold(entries, thershold_at_percentile=.80):
    update_times = []
    for entry_sr in range(len(entries)):
        update_times.append(entries[entry_sr]['update_time']['$date'])
    update_time_ar = np.asarray(update_times)
    return  np.quantile(update_time_ar, thershold_at_percentile)

def write_corpora(fname_train, fname_test, entries, update_time_80_percentile):
    with open(fname_train, 'w') as f_train, open(fname_test, 'w') as f_test:
        for entry_sr in range(len(entries)):
            update_time =  entries[entry_sr]['update_time']['$date']
            
            if update_time < update_time_80_percentile:
                f = f_train
            else:
                f = f_test
            
            edges_snaps = []
            for link_key in entries[entry_sr]['link_map'].keys():
                src_id = entries[entry_sr]['link_map'][link_key]['src_id']
                dst_id = entries[entry_sr]['link_map'][link_key]['dst_id']
                # src_class_id = entries[entry_sr]['snap_map'][src_id]['class_id']
                # dst_class_id = entries[entry_sr]['snap_map'][dst_id]['class_id']
                edges_snaps.append((src_id,dst_id))

            graph_dict = build_graph(edges_snaps)
            paths = all_paths(graph_dict)

            for key in paths:
                for path in paths[key]:
                    paths_str = ""
                    snap_node = path[0]
                    snap_node_cid = entries[entry_sr]['snap_map'][snap_node]['class_id']
                    assert snap_node_cid[0:20] == "com-snaplogic-snaps-", snap_node_cid
                    paths_str += (snap_node_cid[20:])
                    for snap_node in path[1:]:
                        snap_node_cid = entries[entry_sr]['snap_map'][snap_node]['class_id']
                        assert snap_node_cid[0:20] == "com-snaplogic-snaps-",  snap_node_cid
                        paths_str += (" "+ snap_node_cid[20:])
                    
                    f.write(paths_str + "\n")

if __name__ == "__main__":

    json_file = "data/snaplogic_pipeline_catalog.json"
    train_file = "corpora/snaps_corpa_train.txt"
    test_file = "corpora/snaps_corpa_test.txt"
    entries = get_from_json(json_file, "entries")
    update_time_80_percentile = get_update_time_split_thershold(entries, 0.80)
    write_corpora(train_file, test_file,entries, update_time_80_percentile)
