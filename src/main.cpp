#include "Fisher_Kolmogorov.hpp"


int main(int argc, char* argv[]){
    
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int degree = 1;

    double T = 1;
    double deltat = 0.05;
    Fisher_Kolmogorov problem("..\brain_mesh.msh");
    problem.setup();
    problem.solve();
    
    return 0;
}