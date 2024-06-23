/*
#include "Fisher_Kolmogorov.hpp"


int main(int argc, char* argv[]){
    
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int degree = 1;

    double T = 1;
    double deltat = 0.01;

    Fisher_Kolmogorov problem("../mesh/mesh-square-40.msh", degree, T, deltat);
    problem.setup();
    problem.solve();
    
    return 0;
}
*/
#include "Fisher1D.hpp"


int main(int argc, char* argv[]){
    
    /*
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int degree = 1;

    double T = 1;
    double deltat = 0.01;

    Fisher_Kolmogorov problem("../mesh/mesh-square-40.msh", degree, T, deltat);
    problem.setup();
    problem.solve();
    */

    const unsigned int degree = 1;
    double T = 1;
    double deltat = 0.5;
    const unsigned int N = 39; 

    Fisher1D problem("../mesh/mesh-square-40.msh", N ,degree, T, deltat);
    problem.setup();
    problem.solve();
    
    return 0;
}