"""Data preparation scripts to turn audio files into tfrecord for faster
training"""
import os
import json
import tensorflow as tf

import datahelper.dataset as ds

DEGRADE_MAP = {'downsample': ds.downsample_by,
              }


#paths
tf.app.flags.DEFINE_string('dataroot',
                           './data',
                           """Root dir of where to find all audio. If a text
                           file is given, all paths in the text file is
                           relative to here.""")
tf.app.flags.DEFINE_string('output_dir',
                           './tfrecord_datasets',
                           "Output path")
tf.app.flags.DEFINE_string('fileslist',
                           None,
                           """List of files to be read in and preprocessed""")

#Signal processing params
tf.app.flags.DEFINE_float('silence_thresh',
                          60,
                          """Silence threshold. Negative values means no
                          thresholding""")
tf.app.flags.DEFINE_integer('seq_length',
                            8192,
                            """Sequence length in number of samples""")
tf.app.flags.DEFINE_integer("gt_rate",
                            16000,
                            """Groundtruth sampling rate. Set to None for
                            native rate""")
tf.app.flags.DEFINE_enum('degrade_type',
                         'downsample', DEGRADE_MAP.keys(),
                         """How to degrade the original signal"""
                        )
tf.app.flags.DEFINE_integer('degrade_args',
                            2,
                            """Args to pass to degrade function""")
tf.app.flags.DEFINE_integer('segs_per_sample',
                            10,
                            """Segments per sample""")
#Additional metadata
tf.app.flags.DEFINE_string('txt_desc',
                           None,
                           """Some description to be added to metadata file""")
tf.app.flags.DEFINE_string('datasetname',
                           None,
                           """What to name the dataset. If none, the name of
                           the text file is used""")
tf.app.flags.DEFINE_string("source_name",
                           None,
                           """Name of the data source, eg VCTK""")

FLAGS = tf.app.flags.FLAGS

def main(argv):
    """Preprocessing script for audio"""

    if FLAGS.datasetname:
        dset_basename = FLAGS.datasetname
    else:
        dset_basename = os.path.splitext(os.path.basename(FLAGS.fileslist))[0]

    dataset_metadata = {'source_name': FLAGS.source_name,
                        'source_txt_file': os.path.basename(FLAGS.fileslist),
                        'degrade_function': FLAGS.degrade_type,
                        'degrade_params': FLAGS.degrade_args,
                        'groundtruth_sample_rate': FLAGS.gt_rate,
                        'sequence_length': FLAGS.seq_length,
                        'silence_thresh': FLAGS.silence_thresh,
                        'description': FLAGS.txt_desc,
                       }
    dataset_identifier = '-'.join(map(str, [dset_basename,
                                            FLAGS.gt_rate,
                                            FLAGS.degrade_type,
                                            FLAGS.degrade_args,
                                           ])
                                 )
    dataset_metadata['dataset_identifier'] = dataset_identifier
    output_filename = dataset_identifier + '.tfrecord'
    meta_filename = dataset_identifier + '.json'
    dataset_metadata['tfrecord_files'] = [output_filename]

    output_fullpath = os.path.join(FLAGS.output_dir, output_filename)
    meta_fullpath = os.path.join(FLAGS.output_dir, meta_filename)

    print("---------------------------------------------------")
    print(f"Preparing {FLAGS.fileslist} into {output_filename}")
    print("---------------------------------------------------")
    print("Config:")
    print(json.dumps(dataset_metadata, indent=2))

    dset = ds.audio_dataset_from_fileslist(FLAGS.fileslist,
                                           FLAGS.dataroot,
                                           trim_silence=FLAGS.silence_thresh,
                                           gt_rate=FLAGS.gt_rate,
                                          )
    dset = ds.get_segment_dataset(dset,
                                  length=FLAGS.seq_length,
                                  segs_per_sample=FLAGS.segs_per_sample,
                                 )
    degrade_fn = lambda x: DEGRADE_MAP[FLAGS.degrade_type](x, FLAGS.degrade_args)
    dset = ds.get_lq_hq_pair(dset,
                             degrade_fn,
                            )

    serialize_dset = dset.map(ds.tf_serialize_example)

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MkDir(FLAGS.output_dir)

    writer = tf.data.experimental.TFRecordWriter(output_fullpath)
    print("---------------------------------------------------")


    with tf.Session() as sess:
        print("--------------Session starting---------------------")
        sess.run(writer.write(serialize_dset))
        print('Done. Writing metadata')
        with tf.gfile.Open(meta_fullpath, 'w') as f:
            f.write(json.dumps(dataset_metadata, indent=4))


if __name__ == '__main__':
    tf.app.run(main)
