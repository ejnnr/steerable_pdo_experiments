from diffop_experiments import MNISTRotModule

def process_dataset(module):
    module.setup("fit")
    loader = module.train_dataloader()

    # See https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean:", mean.item())
    print("Std:", std.item())

if __name__ == "__main__":
    process_dataset(MNISTRotModule(batch_size=128, validation_size=0, normalize=False, upsample=False, pad=False))

    # To check whether the normalization is implemented correctly, you can instead comment out these lines
    # Mean should be close to 0 and std close to 1

    # process_dataset(MNISTRotModule(batch_size=128, validation_size=0, normalize=True, upsample=False, pad=False))