import numpy as np
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    index_u_g = data['u'] - data['g']
    index_r_i = data['r'] - data['i']

    # Make a redshift array
    redshift = data['redshift']

    # Create the plot with plt.scatter and plt.colorbar
    plot = plt.scatter(index_u_g, index_r_i, s=0.8, lw=0, c=redshift, cmap=cmap)
    cb = plt.colorbar(plot)
    cb.set_label('Redshift')
    
    # Define your axis labels and plot title
    plt.xlabel('Index u-g')
    plt.ylabel('Index r-i')
    plt.title('Redshift u-g v.s. r-i')
      
    # Set any axis limits
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1)

    plt.show()
