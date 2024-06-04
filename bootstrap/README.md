# Notions

## Mean image

- Mean image is the average of a set of images. 
- For a given digit in the MNIST dataset, the mean image is computed by averaging the pixel values of all images of that digit. 
- The steps to compute a mean image are:
    1. Select all images of a specific digit
    2. Compute the pixel-wise average (For each pixel position, compute the average value of that pixel across all selected images)

This results in an image where each pixel value represents the average of that pixel's values in the original images. Mean images are useful for visualizing the general shape and features of each digit as seen by the model


## Plot and subplot

Plot and Subplot are terms commonly used in data visualization, particularly with libraries like Matplotlib

### Plot

A plot is a graphical representation of data. In the context of Matplotlib:
- A single plot can be created using functions like `plt.plot()`, `plt.imshow()` or `plt.bar()``
- Plots can display data in various forms such as line charts, bar charts, images, etc.

### Subplot

- Subplot is a smaller plot within a larger figure that contains multiple plots
- Subplots allow to arange multiple plots in a grid within the same figure, making it easier to compare the side by side
- Subplots are created by using `plt.subplot()` or `plt.subplots()`
