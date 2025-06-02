import json
import os


if __name__ == '__main__':
    
    unique_paths = set()
    path_dict = {}
    special_paths = {'lung_body_regions_localization', 'lung_body_regions_attributes', 'lung_body_regions_degree', 
                'trachea_body_regions_attributes', 'trachea_body_regions_degree',
                'pleura_body_regions_localization', 'pleura_body_regions_attributes','pleura_body_regions_degree'}
    
    for mode in ['train', 'val', 'test']:
        reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        # all images corresponding to reports in radrestruct/{split}_qa_pairs
        for report in reports:
            with open(f'data/radrestruct/{mode}_qa_pairs/{report}', 'r') as f:
                qa_pairs = json.load(f)
                
                for qa_pair in qa_pairs:
                    info = qa_pair[3]
                    base_path = info['path']
                    path_options = info['options']
                    for option in path_options:
                        path = f"{base_path}_{option}"
                        kb_path = path
                        
                        if path in special_paths:
                            path_elements = path.split("_")
                            last = path_elements[-1]
                            without_last = path_elements[0:-1]
                            without_last.append(path_elements[0])
                            without_last.append(last)
                    
                    #path_elements.append(path_elements[0])
                            kb_path = f"{'_'.join(without_last)}_{option}"
                        kb_path = kb_path.replace('body_region', 'body region')
                        kb_path = kb_path.replace('body_regions', 'body regions')
                        unique_paths.add(path)
                        path_dict[(base_path,option)] = kb_path
                    
                    
    print(len(unique_paths))
    print(len(path_dict))
    print(len(path_dict.values()))                
    print(unique_paths)
    print(path_dict)
    
    with open("path_lookup.json", "w") as f:
        json.dump(path_dict, f)