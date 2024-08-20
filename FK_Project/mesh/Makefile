# Define the Gmsh executable
GMSH = gmsh

# Define the target .msh file
TARGET_3D = brain-h3.0.msh

# Define the source .geo file
SOURCE_3D = brain-h3.0.geo

# Default rule to generate .msh from .geo
all: $(TARGET_3D)

$(TARGET_3D): $(SOURCE_3D)
	$(GMSH) -3 $(SOURCE_3D) -o $(TARGET_3D)

# Clean rule to remove generated .msh files
clean:
	rm -f $(TARGET_3D)