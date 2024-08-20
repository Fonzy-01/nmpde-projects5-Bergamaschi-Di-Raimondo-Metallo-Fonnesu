#include "Fisher_Kolmogorov.hpp"

int main(int argc, char* argv[]){
    
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int degree = 1;

    double T = 2;
    double deltat = 0.1;

    Fisher_Kolmogorov problem("../mesh/brain-h3.0.msh", degree, T, deltat);
    problem.setup();
    problem.solve();
    
    return 0;
}
