#ifndef FISHER_KOLMOGOROV_HPP
#define FISHER_KOLMOGOROV_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class Fisher_Kolmogorov
{
public:

    // Physical dimension (1D, 2D, 3D)
    static constexpr unsigned int dim = 3;

    // Function describing the value of the spreading coefficient -> the bigger alpha is, the faster is the spreading
    class FunctionAlpha : public Function<dim> 
    {
    public : 
        virtual double 
        value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
        {   
            return 1.0;
        }
    }; 

    // Function describing the behaviour of D
    class FunctionD : public Function<dim> 
    {
    public : 
        virtual void 
        matrix_value(const Point<dim> & /*p*/,
            FullMatrix<double> &values) const /*override*/
        {   
            // Here go through the diagonal the values for the extracellular diffusion
            for(unsigned int i = 0; i < dim; i++ ){
                for(unsigned int j = 0; j < dim; j++){
                    if(i == j){
                        //da ricontrollare 
                        values(i,j) = 0.1; 
                    }
                    else{
                        values(i,j) = 0.0; 
                    }
                }
            }

            // Here should go the values for the d_axn
            // Example: values(1,1) += 0.05;
            values(1,1) += 10.0;
        }

        virtual double 
        value(const Point<dim> &/*p*/, const unsigned int component1 = 0, const unsigned int component2 = 1)  const /*override*/ 
        {
            return (component1 == component2) ? 1.0 : 0.0;
        }
    }; 

    // Function for the initial concentration in a specific region of the mesh
    class FunctionC0 : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const override
        {
            // Point<dim> origin;
            Point<dim> starting_point(0.0, 0.0, 0.0);
            double max_value = 0.95;
            double std_dev = 0.0333;

            if(p.distance(starting_point) <= 0.3)
            {
                return max_value
                        * std::exp(-(p.distance(starting_point) * p.distance(starting_point)) / (2 * std_dev * std_dev));
            }
            
            return 0;
        }
    };

    // Function for the forcing term.
    class ForcingTerm : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
    };

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    Fisher_Kolmogorov(const std::string &mesh_file_name_,
                const unsigned int &r_,
                const double &T_,
                const double &deltat_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(T_)
        , mesh_file_name(mesh_file_name_)
        , r(r_)
        , deltat(deltat_)
        , mesh(MPI_COMM_WORLD)
    {}

    // Initialization.
    void
    setup();

    // Solve the problem.
    void
    solve();

protected:
    // Assemble the tangent problem.
    void
    assemble_system();

    // Solve the linear system associated to the tangent problem.
    void
    solve_linear_system();

    // Solve the problem for one time step using Newton's method.
    void
    solve_newton();

    // Output.
    void
    output(const unsigned int &time_step) const;

    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    //Problem definition. ///////////////////////////////////////////////////////////
    
    FunctionD D;

    FunctionAlpha alpha;

    // Forcing term.
    ForcingTerm forcing_term;

    // Initial conditions.
    FunctionC0 c_0;

    // Current time.
    double time;

    // Final time.
    const double T;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // Jacobian matrix.
    TrilinosWrappers::SparseMatrix jacobian_matrix;

    // Residual vector.
    TrilinosWrappers::MPI::Vector residual_vector;

    // Increment of the solution between Newton iterations.
    TrilinosWrappers::MPI::Vector delta_owned;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution;

    // System solution at previous time step.
    TrilinosWrappers::MPI::Vector solution_old;
};

#endif 