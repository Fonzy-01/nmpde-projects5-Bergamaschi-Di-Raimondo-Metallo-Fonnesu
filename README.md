# Fisher-Kolmogorov equation for neurodegenerative diseases
This project aims at numerically solve, through a finite element solver, the Fisher-Kolmogorov equation, with a focus on applications in the field of neurodegenerative diseases.

## Repository organization
The main folder of this repository is the FK_Project folder. Inside that we can find two other folders, mesh, containing all the necessary files to create the .msh file of the brain, and src.
### Src
This folder contains six different files:
- Fisher_Kolmogorov.cpp: contains all the functions needed to assemble and solve the 3D problem.
- Fisher_Kolmogorov.hpp: header file containing the declaration of classes, functions and variables for the 3D case.
- main.cpp: file that controls the execution of the 3D solver.
- Fisher1D.cpp: contains all the functions needed to assemble and solve the 1D problem.
- Fisher1D.hpp: header file containing the declaration of classes, functions and variables for the 1D case.
- main1D.cpp: file that controls the execution of the 1D solver.

## Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```

### Creation of the mesh
The 1D mesh is directly created through the code, while the 3D mesh is obtained in the following way:
```bash
$ cd FK_Project
$ cd mesh
$ make
```

### Creation of the executables
Then run the following commands:
```bash
$ cd FK_Project
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Executables and results

There will be two executables, one for the 1D case and the other for the brain case.
The former can be executed through:
```bash
$ ./FK_Project_1D
```
while the latter with:
```bash
$ ./FK_Project
```

### Output
The program creates two different types of file: the first ones are the .vtk files, related to the 1D case, while the others are .vtu files, that contains results for the second case. These files can be opened with Paraview.