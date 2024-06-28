#include "Fisher1D.hpp"

void
Fisher1D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh for the 1D case.
  {
    std::cout << "Initializing the mesh" << std::endl;
    GridGenerator::subdivided_hyper_cube(mesh, N + 1, -1.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

   
    fe = std::make_unique<FE_Q<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

  
    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;


    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // We use the sparsity pattern to initialize the system matrix
    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    // We initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs()); 
    solution_owned.reinit(dof_handler.n_dofs()); 
    delta_owned.reinit(dof_handler.n_dofs()); 
  }
}


void 
Fisher1D::assemble_system()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  residual_vector = 0.0;

  // Value and gradient of the solution on the cell.
  std::vector<double> solution_loc(n_q);
  std::vector<Tensor<1, dim>> derivative_loc(n_q);

  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);

  forcing_term.set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_residual = 0.0;

    fe_values.get_function_values(solution, solution_loc);
    fe_values.get_function_gradients(solution, derivative_loc);
    fe_values.get_function_values(solution_old, solution_old_loc);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // Evaluate coefficients on this quadrature node.
      const double alpha_loc = alpha.value(fe_values.quadrature_point(q));

      // D is a scalar in 1D, so no need for a matrix.
      const double D_loc = D.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          // Mass matrix/delta  
          cell_matrix(i, j) += fe_values.shape_value(i, q) *
                              fe_values.shape_value(j, q) / deltat *
                              fe_values.JxW(q);

          // First term of the stiffness matrix
          cell_matrix(i, j) += D_loc * fe_values.shape_grad(j, q) *
                              fe_values.shape_grad(i, q) *
                              fe_values.JxW(q);

          // Second term of the stiffness matrix
          cell_matrix(i, j) -= (alpha_loc - 2.0 * alpha_loc * solution_loc[q]) * 
                                fe_values.shape_value(j, q) *
                                fe_values.shape_value(i, q) *
                                fe_values.JxW(q);
        }

        // Assemble the residual vector (with changed sign).

        // Time derivative term.
        cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                            deltat * fe_values.shape_value(i, q) *
                            fe_values.JxW(q);

        // Diffusion term.
        cell_residual(i) -= D_loc * derivative_loc[q] *
                            fe_values.shape_grad(i, q) *
                            fe_values.JxW(q);

        // Non linear term.
        cell_residual(i) += alpha_loc *
                            solution_loc[q] *
                            (1 - solution_loc[q]) * 
                            fe_values.shape_value(i, q) *
                            fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);

    system_matrix.add(dof_indices, cell_matrix);
    residual_vector.add(dof_indices, cell_residual);
}

  system_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
}


void 
Fisher1D::solve_newton()
{
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-6;

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
  {
    assemble_system();
    residual_norm = residual_vector.l2_norm();

    std::cout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush << std::endl;

    // We actually solve the system only if the residual is larger than the
    // tolerance.
    if (residual_norm > residual_tolerance)
    {
      solve_linear_system();

      solution_owned += delta_owned;
      solution = solution_owned;
    }
    else
    {
     std::cout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }
}


// da sistemare
void
Fisher1D::solve()
{
  std::cout << "===============================================" << std::endl;  
   std::cout << "===============================================" << std::endl;

    time = 0.0;

    // Apply the initial condition.
    {
        std::cout << "Applying the initial condition" << std::endl;
        VectorTools::interpolate(dof_handler, c_0, solution_owned);
        std::cout << " Sono qui " << std::endl; 
        solution = solution_owned;

        // Output the initial solution.
        output(0);
        std::cout << "-----------------------------------------------" << std::endl;
    }

    unsigned int time_step = 0;

    while (time < T-0.5*deltat)
    {
        time += deltat;
        ++time_step;

        // Store the old solution, so that it is available for assembly.
        solution_old = solution;

        std::cout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
                << std::fixed << time << std::endl;

        // At every time step, we invoke Newton's method to solve the non-linear
        // problem.
        solve_newton();

        output(time_step);

        std::cout << std::endl;
    }
}


void 
Fisher1D::solve_linear_system()
{
    SolverControl solver_control(1000, 1e-12 * residual_vector.l2_norm());

    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(
        system_matrix, PreconditionSSOR<>::AdditionalData(1.0));

    solver.solve(system_matrix, delta_owned, residual_vector, preconditioner);
    std::cout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}


void
Fisher1D::output(const unsigned int &time_step) const
{
  std::cout << "===============================================" << std::endl;

  // Writing the results to a file.
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  data_out.build_patches();

  const std::string output_file_name =
    "output-" + std::to_string(time_step+ 1) + ".vtk";
  std::ofstream output_file(output_file_name);

  // Writing the data to the file in VTK format.
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}
