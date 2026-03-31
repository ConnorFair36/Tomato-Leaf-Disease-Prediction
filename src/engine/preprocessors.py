from torchvision import transforms

DATA_TRANSFORM_DOWNSAMPLE = transforms.Compose([
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    
    transforms.Grayscale(),
    # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    transforms.Resize(size = [150, 150]),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() 
])

DATA_TRANSFORM = transforms.Compose([
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.Grayscale(),
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

