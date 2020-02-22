import tensorflow as tf

def convert2List(value):
    value = np.reshape(value, [-1])
    vl = [int(v) for v in value]
    return vl
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=convert2List(value)))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    value = convert2List(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = bytes(value,'ascii')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_label(label):
    assert len(label) == 68 - 24, '%s' % (len(label))
    label2 = np.array([float(l) for l in label], dtype=np.float32)
    label2[[1, 2, 4, 6, 9]] = np.round(label2[[1, 2, 4, 6, 9]] / 100, decimals=0)
    return label2


def convert_to_TFRecord(args):
    coder = ImageCoder()
    writer = tf.python_io.TFRecordWriter(args.tfrecord_file)
    # dataset, total = dataSet(args, False)
    # for num in range(int(np.ceil(total/args.batch_size))):
    #     image, label = sess
    #     convert_to_example()
    paths, labels = load_data(args)

    for i, (path, label) in enumerate(zip(paths, labels)):
        # path = tf.convert_to_tensor(path)
        # path = bytes(path,encoding='ascii')
        # print(path)
        image_content = tf.gfile.GFile(path,'rb').read()
        image = coder.decode_jpeg(image_content)
        h, w, _ = image.shape
        image = list(np.reshape(image, [-1]))
        img_list = [float(p)/256.0 for p in image]
        # print(type(img_list[0]))
        # print(type(img_list))

        example = convert_to_example(img_list,label,os.path.basename(path),h,w)
        writer.write(example.SerializeToString())
        print('progress %d / %d' % (i, len(paths)))
    writer.close()

def convert_to_example(image, label, name, image_height, image_width):
    image_feature = _float_feature(image)
    label_feature = _float_feature(label)
    name_feature = _bytes_feature(name)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': image_feature,
        'image/label': label_feature,
        'image/filename': name_feature,
        'image/width': _int64_feature(image_width),
        'image?height': _int64_feature(image_height)
    }))
    return example
