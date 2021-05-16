from object_detection.utils import label_map_util, dataset_util
import tensorflow as tf
from PIL import Image
import io
import os


label_map = label_map_util.load_labelmap("label_map.pbtxt")
label_map_dict = label_map_util.get_label_map_dict(label_map)


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def create_tf_example(img_path,
                      annotations_list,
                      image_dir):
    with tf.io.gfile.GFile(os.path.join(image_dir, '{}'.format(img_path)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = img_path.split(".")[0].encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # for index, row in group.object.iterrows():
    for annotation in annotations_list:
        (xmin, ymin, xmax, ymax) = tuple(annotation['bbox'])
        xmins.append(float(xmin) / width)
        xmaxs.append(float(xmax) / width)
        ymins.append(float(ymin) / height)
        ymaxs.append(float(ymax) / height)
        classes_text.append(annotation['category'].encode('utf8'))
        classes.append(class_text_to_int(annotation['category'].lower()))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():

    with tf.io.TFRecordWriter("tmp/img.records") as writer:
        for file in os.listdir("tmp/validation/towel"):
            annotations_list = []
            if file == 'labels':
                continue
            print(f'{file} processing')
            label_path = os.path.join(
                "tmp/validation/towel", "labels", file.replace("jpg", "txt"))
            with open(label_path) as label:
                lines = [line.strip() for line in label.readlines()]
                for line in lines:
                    annotation = {}
                    annotation['category'] = line.split()[0]
                    annotation['bbox'] = line.split()[1:]
                    annotations_list.append(annotation)

            tf_example = create_tf_example(
                file, annotations_list, "tmp/validation/towel")
            writer.write(tf_example.SerializeToString())
            print('Successfully created the TFRecord for file: {}'.format(
                file))

    # writer.close()
    # print('Successfully created the TFRecord file: {}'.format(args.output_path))
    # if args.csv_path is not None:
    #     examples.to_csv(args.csv_path, index=None)
    #     print('Successfully created the CSV file: {}'.format(args.csv_path))
if __name__ == '__main__':
    main()
