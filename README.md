# GAN
A generative adversarial network to generate human facial images.

# Make TFRecords dataset
Reads images from given input directory and creates a TFRecord dataset file that is utilized to train models
```bash
python make_tfrecords.py --image_dir=<celeb_a_images_path \
--out_path=out_dir \
--precropped=False
```

# Train Models
Change configs/gan_config.yml as necessary and train the GAN by issuing the command below
```bash
python train.py --config_path= config_path/gan_config.yml \
--num_epochs=60
```

## Data
Data downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
