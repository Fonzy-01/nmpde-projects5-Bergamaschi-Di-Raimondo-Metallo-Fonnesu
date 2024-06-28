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
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_generator.h>


#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>


#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class Fisher1D
{
public:

    // Physical dimension: for this case, dim = 1
    static constexpr unsigned int dim = 1;

    // Function describing the value of the spreading coefficient -> the bigger alpha is, the faster is the spreading
    class FunctionAlpha : public Function<dim> 
    {
    public : 
        virtual double 
        value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const /*override*/
        {   
            return 2.0;
        }
    }; 

    // Function describing the behaviour of D
    class FunctionD : public Function<dim> 
    {
    public : 
        virtual double 
        value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const /*override*/
        {   
            return 0.0004;
        }

    }; 

    // Function for the initial concentration in a specific region of the mesh
    class FunctionC0 : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> &p,
            const unsigned int /*component*/ = 0) const /*override*/
        {
            //Point<dim> origin;
            Point<dim> starting_point(0.0);
            if(p.distance(starting_point) <= 0.1)
            {
                return 0.1;
            }
            
            return 0;
        }
    };

    // Function for the forcing term.
    class ForcingTerm : public Function<dim>
    {
    public:
        // Constructor.
        ForcingTerm()
        {}

        // Evaluation
        virtual double
        value(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }
    };

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    Fisher1D(const unsigned int &N_, 
                const unsigned int &r_,
                const double &T_,
                const double &deltat_)
        : N(N_)
        , r(r_)
        , T(T_)
        , deltat(deltat_)
    {}

    // Initialization
    void
    setup();

    // Solve the problem
    void
    solve();

protected:
    // Assemble the system
    void
    assemble_system();

    // Solve the linear system associated to the problem
    void
    solve_linear_system();

    // Solve the problem for one time step using Newton's method
    void
    solve_newton();

    // Output
    void
    output(const unsigned int &time_step) const;

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

    // N+1 is the number of elements.
    const unsigned int N;

    // Discretization. ///////////////////////////////////////////////////////////

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;

    // Mesh file name.
    //const std::string mesh_file_name;

    // Mesh.
    Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // Sparsity pattern
    SparsityPattern sparsity_pattern;

    // System matrix
    SparseMatrix<double> system_matrix;

    // System right-hand side
    Vector<double> residual_vector;

    // System solution
    Vector<double> solution;
    Vector<double> solution_old; 
    Vector<double> solution_owned; 
    Vector<double> delta_owned; 
};

#endif 