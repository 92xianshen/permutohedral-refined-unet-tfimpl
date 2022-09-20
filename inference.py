import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.permutohedral_refined_unet import PermutohedralRefinedUNet

theta_alpha, theta_beta, theta_gamma = 80, 0.125, 3
# theta_alpha, theta_beta, theta_gamma = 80, 0.0625, 3
# theta_alpha, theta_beta, theta_gamma = 80, 0.03125, 3
d_bifeats, d_spfeats = 5, 2
bilateral_compat, spatial_compat = 10., 3., 
gt_prob = 0.7
n_iterations = 10

CROP_HEIGHT, CROP_WIDTH = 512, 512
num_bands = 7
num_classes = 4
batch_size = 1
# unet_path = "pretrained/"
unary_generator_path = 'unary_model/'
data_path = "../data/"
save_path = "../result/"

def load_testset(filenames, batch_size=batch_size):
    """Load a tensorflow TFDataset file as a test set"""
    test_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        "x_train": tf.io.FixedLenFeature([], tf.string),
        "y_train": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        x = tf.io.decode_raw(example["x_train"], tf.int32)
        y = tf.io.decode_raw(example["y_train"], tf.uint8)

        x = tf.reshape(x, [512, 512, 7])
        y = tf.reshape(y, [512, 512])

        x = tf.cast(x, tf.float32)
        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-10)
        y = tf.cast(y, tf.int32)

        example["x_train"] = x
        example["y_train"] = y

        return example

    test_dataset = test_dataset.map(_parse_function).batch(
        batch_size, drop_remainder=True
    )

    return test_dataset

def reconstruct(pred_patches):
    """Reconstruct a prediction from the prediction list"""

    def reconstruct_from_patches(
        predictions,
        num_height,
        num_width,
        crop_height=CROP_HEIGHT,
        crop_width=CROP_WIDTH,
    ):
        """Reconstruct from a prediction list. A reverse function of `extract_pairwise_patches`"""

        assert (
            num_height * num_width == predictions.shape[0]
        ), "Dim is wrong. {} X {} != {}".format(
            num_height, num_width, predictions.shape[0]
        )

        prediction = np.ndarray(
            shape=(num_height * crop_height, num_width * crop_width),
            dtype=predictions.dtype,
        )

        for i in range(num_height):
            for j in range(num_width):
                prediction[
                    (crop_height * i) : (crop_height * (i + 1)),
                    (crop_width * j) : (crop_width * (j + 1)),
                ] = predictions[i * num_width + j]

        return prediction

    if pred_patches.shape[0] == 240:
        num_height = 15
        num_width = 16
    elif pred_patches.shape[0] == 256:
        num_height = 16
        num_width = 16
    else:
        print("Prediction shape error!")

    prediction = reconstruct_from_patches(
        pred_patches,
        num_height=num_height,
        num_width=num_width,
        crop_height=CROP_HEIGHT,
        crop_width=CROP_WIDTH,
    )

    # prediction.shape is [num_height x 512, num_width x 512]
    return prediction


def main():
    test_names = os.listdir(data_path)
    print(test_names)
    save_info_name = "rfn.csv"

    # Create model
    model = PermutohedralRefinedUNet(unary_generator_path=unary_generator_path, height=CROP_HEIGHT, width=CROP_WIDTH, n_classes=num_classes, d_bifeats=d_bifeats, d_spfeats=d_spfeats, theta_alpha=theta_alpha, theta_beta=theta_beta, theta_gamma=theta_gamma, bilateral_compat=bilateral_compat, spatial_compat=spatial_compat, gt_prob=gt_prob, n_iterations=n_iterations)

    with open(os.path.join(save_path, save_info_name), "w") as fp:
        fp.writelines("name, theta_alpha, theta_beta, theta_gamma, duration\n")

        for test_name in test_names:
            # Names
            save_npz_name = test_name.replace("train.tfrecords", "rfn.npz")
            save_png_name = test_name.replace("train.tfrecords", "rfn.png")

            # Load one test set
            test_name = [os.path.join(data_path, test_name)]
            test_set = load_testset(test_name, batch_size=1)
            refinements = []

            # Inference
            start = time.time()
            i = 0
            for record in test_set.take(-1):
                print("Patch {}...".format(i))
                
                x, y = record["x_train"], record["y_train"]

                rfn = model(x[0])

                refinements += [rfn.numpy()]

                i += 1

            refinements = np.stack(refinements, axis=0)
            refinement = reconstruct(refinements)
            duration = time.time() - start

            # Save
            np.savez(os.path.join(save_path, save_npz_name), refinement)
            plt.imsave(os.path.join(save_path, save_png_name), refinement, cmap="gray")
            fp.writelines(
                "{}, {}, {}, {}, {}\n".format(
                    test_name, theta_alpha, theta_beta, theta_gamma, duration
                )
            )
            print("{} Done.".format(test_name))


if __name__ == "__main__":
    main()
