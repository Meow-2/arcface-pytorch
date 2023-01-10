
import os


def txt_annotation(dataset_name):
    datasets_path = 'datasets/' + dataset_name
    types_name = os.listdir(datasets_path)
    # 文件名排序
    types_name = sorted(types_name)

    list_file = open('cls_train.txt', 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_file.write(str(cls_id) + ";" + '%s' %
                            (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()
