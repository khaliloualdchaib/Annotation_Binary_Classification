import json


with open('old.json', 'r') as f:
    data = json.load(f)

new_data = {'images': []}

image_id_map = {}
for i, image in enumerate(data['images']):
    image_id_map[image['id']] = i
    annotated = False
    annotations = []
    num_annotations = 0
    for annotation in data['annotations']:
        if annotation['image_id'] == image['id']:
            annotations.append({
                'id': annotation['id'],
                'image_id': i,
                'segmentation': annotation['segmentation'],
                'area': annotation['area'],
                'bbox': annotation['bbox'],
                'iscrowd': annotation['iscrowd'],
                'isbbox': annotation['isbbox'],
                'color': annotation['color']
            })
            num_annotations += 1
            annotated = True
    new_image = {
        'id': i,
        'path': image['path'],
        'width': image['width'],
        'height': image['height'],
        'file_name': image['file_name'],
        'annotated': annotated,
        'annotations': annotations,
        'num_annotations': num_annotations
    }
    new_data['images'].append(new_image)

with open('new.json', 'w') as f:
    json.dump(new_data, f)
