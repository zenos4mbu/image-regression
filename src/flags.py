from absl import flags

flags.DEFINE_string('dataset', 'SINGLE', 'Dataset name.')
flags.DEFINE_string('exp_name', 'SINGLE_sweep', 'Experiment name.')

flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('grid_type', "HASH", 'type of grid.')
flags.DEFINE_string('image_path', "../input/butterfly.jpg", 'Image path.')
flags.DEFINE_string('multiscale_type', "cat", 'How to aggregate features ad different lods.')
flags.DEFINE_integer('num_LOD', 2, 'Levels of detail.')
flags.DEFINE_integer('image_size', 256, 'Input images size.')
flags.DEFINE_integer('n_channels', 3, 'Number of image channels.')
flags.DEFINE_integer('trainset_size', 1, 'Size of the training set.')
flags.DEFINE_integer('feat_dim', 2, ' feat dimension in the grid')
flags.DEFINE_integer('band_width', 8, 'codebook size')
flags.DEFINE_integer('hidden_dim', 16, 'Neural Network Hidden dim')
flags.DEFINE_float('feature_std', 0.01, 'Feature standard deviation')
flags.DEFINE_integer('min_grid_res', 8, 'Minimum grid resolution')
flags.DEFINE_integer('max_grid_res', 64, 'Maximum grid resolution')
flags.DEFINE_integer('grid_lr_factor', 1, 'Learning rate for the grid')


flags.DEFINE_string('mode', 'train', 'train/test/demo.')
flags.DEFINE_integer('mapping_size', 2048, 'Dimension of the input mapping.')
flags.DEFINE_integer('mapping_multiplier', 50, 'Multiplier of the input mapping.')

flags.DEFINE_string('activation', 'RELU', 'RELU/SIN.')
flags.DEFINE_string('argmin', 'soft', 'softhard.')

flags.DEFINE_boolean('display', False,
                     'Display images during training.')
flags.DEFINE_boolean('freeze_nn', False,
                     'Freeze Neural Network.')
flags.DEFINE_boolean('visualize_collisions', False,
                     'Display collisions during training.')

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('lr_kmeans', 0.1, 'Learning rate_kmeans.')
flags.DEFINE_boolean('resume_training', False,
                     'Resume training using a checkpoint.')
flags.DEFINE_boolean('use_grid', True,
                     'Use grid of features')
flags.DEFINE_string('load_checkpoint_dir', '<path>',
                    'Load previous existing checkpoint.')
flags.DEFINE_integer('seed', 42, 'Seed.')
flags.DEFINE_integer('max_epochs', 10000, 'Number of training epochs.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay.')
flags.DEFINE_float(
    'patience', 10, 'Number of epochs with no improvement after which learning rate will be reduced.')
flags.DEFINE_integer('num_workers', 8, 'Number of workers.')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus.')

flags.DEFINE_integer('accumulation', 5, 'Gradient accumulation iterations.')