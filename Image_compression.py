# Import necessary libraries
from matplotlib.image import imread               # Import function for reading images
import numpy as np                                # Import library for numerical computing
import matplotlib.pyplot as plt                   # Import library for plotting
import os                                         # Import library for interacting with operating system

# Set properties of matplotlib figure
plt.rcParams['figure.figsize']=[5,5]               # Set size of figure to 5x5 inches
plt.rcParams.update({'font.size':18})              # Set font size of figure to 18 points

# Load image and convert to grayscale
A = imread(os.path.join('..', 'kitty.jpg'))   # Load image file from directory and store as a 3D array
B = np.mean(A,-1)                                          # Compute the mean of the color channels to convert to grayscale

# Display original image
plt.figure()                                               # Create a new figure
plt.imshow(256-A)                                          # Display the original image
plt.axis('off')                                            # Turn off the axis labels

# Compute SVD of grayscale image and sort magnitudes
Bt = np.fft.fft2(B)                                        # Compute 2D SVD of the image
Btsort = np.sort(np.abs(Bt.reshape(-1)))                   # storing the svd

# Loop over compression ratios and display compressed images
for keep in (0.99, 0.05, 0.01, 0.002):                     # Loop over different compression ratios
    thresh = Btsort[(int(np.floor((1-keep)*len(Btsort))))] #
    ind = np.abs(Bt) > thresh                              #
    Atlow = Bt * ind                                       # 
    plt.figure()                                           # Create a new figure
    plt.imshow(256 - np.real(np.fft.ifft2(Atlow)), cmap='gray') # Display the compressed image
    plt.axis('off')                                        # Turn off the axis labels
    plt.title('Compressed image keeping : ' + str(keep * 100) + '%')  # Set title of figure to compression ratio

# Display 3D surface plot of original grayscale image
from mpl_toolkits.mplot3d import axes3d                   # Import 3D plotting toolkit
plt.rcParams['figure.figsize']=[6,6]                       # Set size of figure to 6x6 inches
fig = plt.figure()                                         # Create a new figure
ax = fig.add_subplot(111,projection='3d')                  # Add a 3D subplot to the figure
X,Y = np.meshgrid(np.arange(1,np.shape(B)[1]+1),np.arange(1,np.shape(B)[0]+1))  # Create a meshgrid for x and y coordinates
ax.plot_surface(X[0::10,0::10],Y[0::10,0::10],256-B[0::10,0::10],cmap='plasma',edgecolor='none')  # Plot 3D surface of image
ax.set_title('Surface plot')                               # Set title of figure
ax.mouse_init()                                            # Initialize mouse controls for 3D plot
ax.view_init(200,270)                                      # Set the initial viewing angle of the plot
plt.show()                                                 # Display the figure
