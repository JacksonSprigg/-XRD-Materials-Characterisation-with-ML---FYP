import matplotlib.pyplot as plt
from ase.db import connect

### Let's look at the simXRD data ###

# I save plots to /scratch. Don't accidentally overload this storage.
#  **READ MONARCH DOCUMENTATION TO MAKE SURE YOU DON'T CRASH THE MACHINE**

# ase.db stores a lot of different atomic data
# See https://wiki.fysik.dtu.dk/ase/ase/db/db.html#description-of-a-row
# I checked the files and will list ALL the USEFUL keys we have access to below.

################## Here is the data that we have access to ###########################
# _keys: Shows some keys
# key_value_pairs: Shows key value pairs
# get: Method that does?
# count_atoms: Method that does?
# toatoms: Method that does?

# I think these are the same?
# chem_form: ex. La2Pd2
# formula: ex. La2Pd2

# intensity: XRD intensities, normalised to 100 | 3501 x 1 vector
# latt_dis: Lattice distances | 3501 x 1 vector
# mass: Atomic mass
# natoms: Number of atoms | int
# numbers: Atomic numbers | int | (N,)
# pbc: Periodic boundary condition flags | bool | (3,) - I think these are all just false and there isn't actually any information here
# simulation_param: Yes - I don't know what this actually means
# symbols: A list, e.g., ['H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'O', 'O', 'O', 'O']
# tager: [Space Group, Crystal System, ??something else??] - I think the something else is bravis lattice types?

# Note the crystal system meaning:
#  1 : Cubic
#  2 : Hexagonal
#  3 : Tetragonal
#  4 : Orthorhombic
#  5 : Trigonal
#  6 : Monoclinic
#  7 : Triclinic
#####################################################################################

## Note that you may need to change this path depending on permissions ##
train_data_path = "/monfs01/projects/ys68/XRD_ML/simXRD_partial_data/train.db"  # Train size = 5000
test_data_path = "/monfs01/projects/ys68/XRD_ML/simXRD_partial_data/test.db"    # Test size  = 2000
val_data_path = "/monfs01/projects/ys68/XRD_ML/simXRD_partial_data/val.db"      # Val size   = 1000
databs = connect(val_data_path)

# Creating an XRD plot and saving it to the below path
image_save_path = "/home/jsprigg/scratch/"
#image_save_path = "/home/jsprigg/ys68/XRD_ML/data_manipulation/example_XRD_plots/"
def plot_xrd_data(latt_dis, intensity, chem_form, atomic_mass, spg, crysystem, bravislatt_type, image_save_path):

    # Plot the X-ray diffraction data
    plt.figure(figsize=(12, 6))
    plt.plot(latt_dis, intensity, 'b-')
    plt.xlabel('Lattice Plane Distance')
    plt.ylabel('Intensity')
    plt.title(f'X-ray Diffraction Pattern for {(chem_form)}')
    plt.grid(True)

    # Get the name of the crystal system
    crystal_systems = {
    1: "Cubic",
    2: "Hexagonal",
    3: "Tetragonal",
    4: "Orthorhombic",
    5: "Trigonal",
    6: "Monoclinic",
    7: "Triclinic"}
    crystal_system_name = crystal_systems.get(crysystem, "Unknown")
    
    # Data information
    plt.text(0.95, 0.95, f'Formula: {chem_form}', transform=plt.gca().transAxes, 
        verticalalignment='top', horizontalalignment='right')
    plt.text(0.95, 0.90, f'Atomic mass: {atomic_mass}', transform=plt.gca().transAxes, 
            verticalalignment='top', horizontalalignment='right')
    plt.text(0.95, 0.75, f'Space Group: {spg}', transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right')
    plt.text(0.95, 0.70, f'Crystal System: {crystal_system_name}', transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right')
    plt.text(0.95, 0.65, f'Bravis Latt Type: {bravislatt_type}', transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right')

    # Save the plot as a PNG file
    plt.savefig(image_save_path+f'xrd_plot_{(chem_form)}.png')

    return

# print values if you want a better idea of what they look like
def looking_at_data(databs, max_iterations, plot=False):
    count = 0
    for row in databs.select():

        # Note, element is a good form for feeding to ML, chem_form is good for reading
        element = getattr(row, 'symbols')

        latt_dis = eval(getattr(row, 'latt_dis'))
        intensity = eval(getattr(row, 'intensity'))

        spg = eval(getattr(row, 'tager'))[0]
        crysystem = eval(getattr(row, 'tager'))[1]
        bravislatt_type = eval(getattr(row, 'tager'))[2]

        chem_form = getattr(row, 'chem_form')
        atomic_mass = getattr(row, 'mass')

        simulation_params = eval(getattr(row, 'simulation_param'))

        # Generate the plot
        if plot:
            plot_xrd_data(latt_dis, intensity, chem_form, atomic_mass, spg, crysystem, bravislatt_type, image_save_path)

        count += 1
        if count == max_iterations:
            break

    return 

# Generates plots
limit = 5       # The amount of XRD rows you want to go through
plot  = True   # This will save the matplots to scratch folder
looking_at_data = looking_at_data(databs, limit, plot=plot)
