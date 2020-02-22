
import tensorflow as tf
def loadTFRecord(args):

    raw_image_dataset = tf.data.TFRecordDataset(args.tfrecord_file)

    def _parse_single(example_proto):
        features = {
            'image/label': tf.FixedLenFeature([68 - 24], tf.float32),
            'image/encoded': tf.FixedLenFeature([112*112*3], tf.float32),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/filename': tf.FixedLenFeature([1], tf.string)
        }
        d = tf.io.parse_single_example(example_proto,features)
        h,w = d['image/height'][0], d['image/width'][0]
        image = tf.reshape(d['image/encoded'], (h,w,3))
        image = tf.image.resize(image,(112,112))
        return (d['image/label'], image, d['image/filename'])
     
    
    da = raw_image_dataset.map(_parse_single)
    da = da.batch(args.batch_size)
    # da = da.make_initializable_iterator()
    da = da.make_one_shot_iterator()
    it = da.get_next()
    return da, it
