import os.path
import hydra
import torch

base_checkpoint_path = "tmp/models/"


def join_name(parts, delimiter="_"):
    parts = [str(p) for p in parts if p is not None and p != ""]
    return delimiter.join(parts)


def checkpoint_chimera(
    model_subdir,
    dataset,
    max_labels_per_class,
    seed,
    generator_features,
    generator_blocks,
    generator_split,
    discriminator_features,
    image_size,
    mix_mode,
    mix_size,
    variant,
    add_file=None,
):

    if mix_mode == "grid":
        noise_mode = "mask_{}x{}".format(mix_size, mix_size)
    elif mix_mode == "segmentation":
        noise_mode = "segmentation"
    else:
        raise ValueError

    parts = [
        "chimera_gen",
        dataset,
        max_labels_per_class,
        seed,
        "G-{}-{}-{}".format(generator_features, generator_blocks, generator_split),
        "D-{}".format(discriminator_features),
        image_size,
        noise_mode,
        variant,
    ]

    checkpoint_path = os.path.join(base_checkpoint_path, model_subdir, join_name(parts))
    if add_file is not None:
        checkpoint_path = os.path.join(checkpoint_path, add_file)

    return checkpoint_path


def checkpoint_cls(
    model_subdir,
    dataset,
    max_labels_per_class,
    seed,
    mix,
    generator_features,
    generator_blocks,
    generator_split,
    discriminator_features,
    image_size,
    mix_mode,
    mix_size,
    variant,
    add_file=None,
):

    parts = [
        "model",
        dataset,
        max_labels_per_class,
        seed,
    ]

    if mix:
        if mix_mode == "grid":
            noise_mode = "mask_{}x{}".format(mix_size, mix_size)
        elif mix_mode == "segmentation":
            noise_mode = "segmentation"
        else:
            raise ValueError

        parts += [
            "chimeramix",
            "G-{}-{}-{}".format(generator_features, generator_blocks, generator_split),
            "D-{}".format(discriminator_features),
            image_size,
            noise_mode,
            variant,
        ]

    checkpoint_path = os.path.join(base_checkpoint_path, model_subdir, join_name(parts))
    if add_file is not None:
        checkpoint_path = os.path.join(checkpoint_path, add_file)

    return checkpoint_path


def check_checkpoint_exists(checkpoint_path, epochs):
    checkpoint_last_path = hydra.utils.to_absolute_path(
        os.path.join(checkpoint_path, "last.ckpt")
    )
    print(checkpoint_last_path)
    if os.path.exists(checkpoint_last_path):
        model = torch.load(checkpoint_last_path)
        checkpoint_epochs = model["epoch"]
        print("checkpoint_epochs", checkpoint_epochs)

        if checkpoint_epochs == epochs:
            return True
        else:
            print("checkpoint exists but not finished. Continue")

    return False
