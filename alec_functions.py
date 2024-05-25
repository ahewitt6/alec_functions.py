import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import sys
#goes back two directories
src_path = "../.."
src_path = "/Users/alechewitt/Desktop/HNL_limits_main"
sys.path.append(src_path)
#src_path_foresee="/Users/alechewitt/Desktop/Git_felix_new/FORESEE"
#sys.path.append(src_path_foresee)
import os
#current_directory = os.getcwd()
#print("Current Directory:", current_directory)
#os.chdir("/Users/alechewitt/Desktop/HNL_Limits_main/Git_felix_new/FORESEE/src")
#current_directory = os.getcwd()
#print("Current Directory:", current_directory)
#from foresee import Utility, Foresee
#os.chdir("/Users/alechewitt/Desktop/HNL_Limits_main/src")
#current_directory = os.getcwd()
#print("Current Directory:", current_directory)
from HNLimits import plot_tools
from HNLimits import hnl_tools
#os.chdir("/Users/alechewitt/Desktop/HNL_Limits_main")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from HNLimits import plot_tools, hnl_tools
import sys
import pandas as pd
from pathlib import Path
import matplotlib

class file_funcs():

    #takes a file path and spits out x and y values of that file
    def extract_vals(file_name,sep = ' '):
        df = pd.read_csv(file_name, header=None,sep=sep)
        mass = np.array(df.iloc[:,0].tolist())
        epsilonsq = np.array(df.iloc[:,1].tolist())
        #replace 0 values with nan, this is to get the interpolation line to go away when it hits 0
        epsilonsq[epsilonsq == 0] = np.nan
        return mass, epsilonsq
    #generalizes the above
    def extract_values(file_path, separator = '\t', column_names=True):
        if column_names == True:
            df = pd.read_csv(file_path, sep=separator)
            var_list = []
            for column in df.columns.tolist():
                var_list.append(df[column].tolist())
            return var_list
        if column_names == False and separator == ',':
            #df = pd.read_csv(file_path, sep=separator)
            df = pd.read_csv(file_path, sep=separator,header=None)
            df = df.drop(df.columns[2],axis=1)
            var_list = []
            for n in range(len(df.columns.tolist())):
                var_list.append(df.iloc[:,n].tolist())
            return var_list
        else:
            print('not implemented yet')
            return None

    #separate the tops from the bottoms
    def obtain_bottoms(files_comp):
        tops = []
        bottoms = []
        for n in range(len(files_comp)):
            top_or = files_comp[n][0][-7:-4]
            if top_or == 'top':
                tops.append(files_comp[n])
            else:
                bottoms.append(files_comp[n])
        return bottoms
    #separate the tops from the bottoms using a list of files
    def obtain_bottoms_file_list(files):
        tops = []
        bottoms = []
        for n in range(len(files)):
            top_or = files[n][-7:-4]
            if top_or == 'top':
                tops.append(files[n])
            if top_or != 'top' and files[n] != '.DS_Store':
                bottoms.append(files[n])
        return bottoms
    
    #takes in a file to extract the values, then interpolates them to another list of x values
    def extract_int_vals(self,file_name, xint, sep = ' '):
        try:
            x, y = self.extract_vals(file_name,sep)
            yint = np.interp(xint, x, y)
        except FileNotFoundError:
            print(file_name, ' not found giving an array of 0s instead.')
            yint = [0 for n in range(len(xint))]
        return(yint)

    #takes a file and plots it
    def plot_file(file_path,xmin=.1,xmax=4,ymin=1e-8,ymax=1e-3):
        x,y = extract_vals(file_path)
        plt.plot(x,y)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.show()

    #takes a file_path which has x, y values that create a contour and returns a dataframe containing the bottom part
    #round basically tells it if  two x values are close enough they are the same point
    def extract_bottom_contour(file_path,round=1):
        df = pd.read_csv(file_path,header = None, sep = ' ')
        df.columns = ['x','y']
        df['x'] = df['x'].round(1)
        df1 = df.groupby('x', as_index=False).min()
        return df1

    #creates a folder if it doesnt already exist
    def create_folder(folder_path):
        folder_path = Path(folder_path)
        # Check if the folder already exists
        if not folder_path.exists():
            # If it doesn't exist, create it
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Folder '{folder_path}' was created.")
        else:
            print(f"Folder '{folder_path}' already exists.")

    def write_to_file(file_name,text_to_write):
        #takes in a path name/file name and something to write to the desired file
        # Open the file in write mode ('w'). This will create it if it doesn't exist.
        with open(file_name, 'a') as file:
            # Write the text to the file
            file.write(f'{text_to_write} \n')



class label_funcs():

    def calculate_angle_at_x(plt,x_val, y_val, x_spec):
        ind = list_funcs.find_closest_index(x_spec,x_val)
        """Calculate the angle at index 'ind' in degrees."""
        if (ind <= 0) or (ind >= len(x_val) - 1):
            raise ValueError("index must be greater than 0 and less than the length of the list - 1.")

        #transform into visual coordinates rather than log coordinates to get the actual angle we see in
        #this corrects for aspect ratios and different scales (e.g. log)
        ax = plt.gca()
        dat = []
        for n in range(len(x_val)):
            dat.append((x_val[n],y_val[n]))
        vis_coords = ax.transData.transform(dat)
        #visual coordinates 
        x_vis = []
        y_vis = []
        for n in range(len(vis_coords)):
            x_vis.append(vis_coords[n][0])
            y_vis.append(vis_coords[n][1])
        ##transforming the point we care about
        ##x_spec_vis,y_spec_vis = ax.transData.transform(x_val[ind],y_val[ind])

        #find the angle in vis coords
        vector_a = (x_vis[ind] - x_vis[ind - 1], y_vis[ind] - y_vis[ind - 1])
        #vector_b = (x_val[ind + 1] - x_val[ind], y_val[ind + 1] - y_val[ind])
        vector_b = (1,0)

        # Calculate the dot product and magnitude of vectors
        dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        magnitude_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
        magnitude_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

        # Ensure not to divide by zero (if magnitude is 0 it means points are the same)
        if magnitude_a == 0 or magnitude_b == 0:
            raise ValueError("Adjacent points cannot be the same.")

        # Compute the angle in radians and then convert to degrees
        angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))
        angle_degrees = math.degrees(angle_radians)


        #correct for aspect ratio
        #obtain the aspect ratio
        # Obtain the range for the x-axis and y-axis
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        # Obtain the size of the axes in pixels
        bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        ax_width, ax_height = bbox.width, bbox.height

        # Calculate the aspect ratio of the axes (data units per inch)
        aspect_ratio = (y_range / ax_height) / (x_range / ax_width)
        print(aspect_ratio)

        #correct angle 
        #aspect_ratio = 1
        #cant get aspect ratio to work right
        angle_degrees = np.arctan(np.tan(angle_degrees*np.pi/180)/aspect_ratio)*(180/np.pi)
        
        return angle_degrees

    #finds a good label placement on a curve and within some window (i.e. bounds of the plot)
    #x_min, x_max, y_min, y_max: defines the window that the label must lie
    #nsteps: how much to shift the label each time it is outside of the parameters (too close to the curve or too close to the boundaries)
    def find_label_placement(plt,x_vals, y_vals, threshold=0.1,x_min = 0.1, x_max = 4, y_min=1e-8,y_max=1e-3,nstep=5):
        #filter x_vals and y_vals to be within the plot window
        x_vals, y_vals = list_funcs.filter_points_window(x_vals, y_vals, x_min, x_max, y_min,y_max)
        if len(x_vals)!=0:
            print('ya')
            # Calculate the centroid of the data points as initial label placement guess
            centroid_x = np.mean(x_vals)
            centroid_y = np.mean(y_vals)
            print(centroid_y)
            
            distances = np.sqrt((x_vals - centroid_x)**2 + (y_vals - centroid_y)**2)
            sorted_indices = np.argsort(distances)  # Sort points by distance from centroid

            # Iterate over points starting from the closest one to the centroid
            #for index in sorted_indices:
            index = sorted_indices[0]
            candidate_x = x_vals[index]
            candidate_y = y_vals[index]
            is_far = True
            
            # Check if the candidate point is far enough from all other points
            #for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            #    if i == index:  # Skip the candidate point itself
            #        continue
            x = np.mean(x_vals)
            y = np.mean(y_vals)
            distance = np.sqrt((candidate_x - x)**2 + (candidate_y - y)**2)
            while distance < threshold*centroid_y:
                #candidate_y = candidate_y - (y_max-y_min)/nstep
                candidate_y = candidate_y*1.1
                is_far = False  # Not a good placement if too close to another point
                #break
                distance = np.sqrt((candidate_x - x)**2 + (candidate_y - y)**2)
            if distance > threshold:
                is_far=True
            
            # If the point is not too close to any other point, use it for label placement
            if is_far:
                label_x = candidate_x
                label_y = candidate_y
                # Angle can be determined here based on the dataset, set to 0 for simplicity
                label_angle = label_funcs.calculate_angle_at_x(plt,x_vals, y_vals, label_x)
                return candidate_x, candidate_y, label_angle

        # If no suitable placement is found, return None or center as default
        #return centroid_x, centroid_y, 0

        else:
            print('Didnt work')
            return x_min, y_min, 45*0


class list_funcs():

    #takes a list of x values and y values, x_vals, and y_vals and gets rid of all points that are outside of the window
    def filter_points_window(x_vals, y_vals, x_min = 0.1, x_max = 4, y_min=1e-8,y_max=1e-3):
        x_ref = []
        y_ref = []
        for n in range(len(x_vals)):
            if x_vals[n]>= x_min and x_vals[n]<=x_max and y_vals[n]>= y_min and y_vals[n]<=y_max:
                x_ref.append(x_vals[n])
                y_ref.append(y_vals[n])
        return x_ref, y_ref

    #takes x_spec (a number), x_val (a list) and finds the index of x_val that has a value closest to x_spec
    def find_closest_index(x_spec, x_val):
        """Find the index of the value in x_val that is closest to x_spec."""
        # Calculate the absolute difference between x_spec and all x_val elements
        differences = [abs(x - x_spec) for x in x_val]
        # Find the index of the smallest difference
        closest_index = differences.index(min(differences))
        return closest_index

    #takes two lists (i.e. a set of points) finds nan values in the y array and gets rid of it and the corresponding x value
    def ditch_nan_values(x,y):
        x_new = []
        y_new = []
        for n in range(len(x)):
            if math.isnan(x[n]) == False:
                x_new.append(x[n])
                y_new.append(y[n])
        return(x_new, y_new)

    #takes x, y, z lists and eliminates a desired value in z and corresponding elements in x and y
    def eliminate_value(x,y,z, eliminate_value = 0):
        # Use list comprehension to filter the elements
        filtered_x = [x_val for x_val, z_val in zip(x, z) if z_val != eliminate_value]
        filtered_y = [y_val for y_val, z_val in zip(y, z) if z_val != eliminate_value]
        filtered_z = [z_val for z_val in z if z_val != eliminate_value]
        return filtered_x, filtered_y, filtered_z

    #takes in 3 arrays and returns an array of the max value between the 3, entry by entry
    def max_arr(arr1,arr2,arr3):
        arrs = [arr1,arr2,arr3]
        max_array = np.maximum.reduce(arrs)
        return max_array

    #save 2 lists to a dataframe and save to a txt file
    #folder_name, path to folder
    #file (e.g. bla.txt)
    def save_x_y_to_txt(x,y,folder_name,file):
        mass = x
        epsilonsq = y
        #create directory if it doesnt exist
        # Check if the directory does not exist
        if not os.path.exists(folder_name):
            # Create the directory
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' has been created. Saving files")
            # Create the DataFrame using the dictionary
            df = pd.DataFrame(np.array([mass,epsilonsq]).T)
            file_path = folder_name + file 
            df.to_csv(file_path, header=None, sep=' ', index=False)
        else:
            print(f"Folder '{folder_name}' already exists. Saving files")
            # Create the DataFrame using the dictionary
            df = pd.DataFrame(np.array([mass,epsilonsq]).T)
            file_path = folder_name + file 
            df.to_csv(file_path, header=None, sep=' ', index=False)

    #makes an actual copy of a list and can be modified without affecting the original
    def copy(listt):
        list_copy = copy.deepcopy(listt)
        return list_copy

#useful constants
class const():
    #fermi's constant
    def Gf():
        GF = 1.1663788 * (10**-5) 
        return GF

    #speed of light
    def c():
        c = 299792458 
        return c
    
    #convert s to 1/GeV (check this)
    #1 s = 1/(6.582119569 * 10**-25) 
    def SectoGev():
        SectoGeV = 1/(6.582119569 * 10**-25) 
        return SectoGeV

    #CKM matrix elements
    def VH(element):
        if element=='Vud': return 0.97373
        elif element=='Vus': return 0.2243
        elif element=='Vcs': return 0.975
        elif element=='Vcd': return 0.221
        elif element=='Vcb': return 40.8E-3
        elif element=='Vub': return 3.82E-3
        elif element=='Vtd': return None
        elif element=='Vts': return None
        elif element=='Vtb': return None

class plot_funcs():
    def create_spectra(prod_info,var,ylabel = r"$E_N$ (GeV)",save_fig=False,folder = None, file_name=None):
        #creates spectra for x,y,z inputs
        channels = list(prod_info.keys())
        x = []; y = []; z = []
        for channel in channels:
            for n in range(len(prod_info[channel][var])):
                x.append(prod_info[channel][var][n].theta())
                y.append(prod_info[channel][var][n].e)
                z.append(prod_info[channel]['ws'][n])
        foresee.make_spectrumplot(np.log10(x), np.log10(y),np.array(z), prange=[[-5, 0, 120],[ 0, 4, 50]],vmin=None,vmax=None)
        plt.xlabel(r'$\theta$ (rad)')
        plt.ylabel(ylabel)
        if save_fig == True:
            af.file_funcs.create_folder(folder)
            plt.savefig(folder + file_name,dpi=300)
        else:
            pass
    #ranges are of form [[xm,xM],[ym,yM],[zm,zM]]
    def create_2d_hist(x,y,z, val_ranges = None, bins = [120,50],log = False):
        if val_ranges == None:
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            zmin = min(z)
            zmax = max(z)
        if val_ranges != None:
            xmin, xmax = val_ranges[0]
            ymin, ymax = val_ranges[1]
            zmin, zmax = val_ranges[2]
        matplotlib.rcParams.update({'font.size': 15})
        fig = plt.figure(figsize=(7,5.5))
        ax = plt.subplot(1,1,1)
        h=ax.hist2d(x=x,y=y,weights=z,
                    bins=bins,range=[[xmin,xmax],[ymin,ymax]],
                    norm=matplotlib.colors.LogNorm(vmin=zmin, vmax=zmax), cmap="rainbow",
        )
        fig.colorbar(h[3], ax=ax)

        #ax.set_xticks([1, 10])
        #ax.set_xticklabels([r'$10^0',r'$10$'])

        ax.set_xlabel(r"x (m)")
        ax.set_ylabel(r"y (m)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if log == True:
            ax.set_xscale('log')
            ax.set_yscale('log')

        return h,ax

    def extract_vals_2dhist(h):
        #extracts values of a h object (plt, h = ax.hist2d())
        H, xedges, yedges, image = h
        # Extracting values
        nx, ny = H.shape
        xs = []; ys = []; zs = []
        for ix in range(nx):
            for iy in range(ny):
                # Calculating bin centers from edges
                x_center = 0.5 * (xedges[ix] + xedges[ix+1])
                y_center = 0.5 * (yedges[iy] + yedges[iy+1])
                z_value = H[ix, iy]  # Bin height
                print(f"Bin center (x,y): ({x_center:.2f}, {y_center:.2f}), Height (z): {z_value}")
                xs.append(x_center)
                ys.append(y_center)
                zs.append(z_value)
        return xs, ys, zs
