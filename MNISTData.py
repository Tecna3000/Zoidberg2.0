import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

import numpy as np 
import matplotlib.pyplot as plt
from keras.src.datasets import mnist

class MNISTData:
    def __init__(self):

        # Load the dataset
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

        # Ensure the dataset shapes are correct
        assert self.train_images.shape == (60000, 28, 28)
        assert self.test_images.shape == (10000, 28, 28)
        assert len(self.train_labels) == 60000
        assert len(self.test_labels) == 10000

    def display_statistics(self):

        # Display basic statistics
        unique, counts = np.unique(self.train_labels, return_counts=True)
        
        print("Distribution of each digit in the training set:")
        for digit, count in zip(unique, counts):
            print(f"Digit {digit}: {count} samples")
        plt.bar(unique, counts)
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.title('Distribution of Digits in Training Set')
        plt.show()

    def show_images(self, dataset='train', index=0):
        # Display one desires image from a chosen dataset
        if dataset == 'train':
            image = self.train_images[index]
            label = self.train_labels[index]
        elif dataset == 'test':
            image = self.test_images[index]
            label = self.test_labels[index]
        else:
            raise ValueError("Dataset must be 'train' or 'test'")
        
        plt.imshow(image, cmap='gray')
        plt.title(f'Digit: {label}')
        plt.show()

    def compute_mean_images(self):
        # Compute the mean image for each digit
        mean_images = {}
        for digit in range(10):
            indices = np.where(self.train_labels == digit)
            mean_images[digit] = np.mean(self.train_images[indices], axis=0)
        return mean_images
    
    def display_mean_images(self):
        # Display the mean images for each digit
        mean_images = self.compute_mean_images()
        plt.figure(figsize=(10, 5))
        for digit, mean_image in mean_images.items():
            plt.subplot(2, 5, digit + 1)
            plt.imshow(mean_image, cmap='gray')
            plt.title(f'Digit: {digit}')
            plt.axis('off')
        plt.suptitle('Mean Images of Each Digit')
        plt.show() 

    def reshape_images(self):
        # Reshape the images datasets to size [n, 28*28] = [n, 784]
        self.train_images_flat = self.train_images.reshape((60000, 28 * 28))
        self.test_images_flat = self.test_images.reshape((10000, 28 * 28))
        print(f"Reshaped training images shape: {self.train_images_flat.shape}")
        print(f"Reshaped test images shape: {self.test_images_flat.shape}")


mnist_data = MNISTData()
mnist_data.display_statistics()
mnist_data.show_images(dataset='train', index=10)
mnist_data.display_mean_images()
mnist_data.reshape_images()