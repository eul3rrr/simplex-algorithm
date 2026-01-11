import numpy as np
epsilon = 1e-10       # Not sure what im doing with this line, if it works dont fix it

cost = np.array([-10, 12, 3, 0, 0, 0])          # Minimize cost^T*x subject to
constraints = np.array([[1, 2, 2, 1, 0, 0]      # Ax = b, x >= 0 where A is a m*n matrix
                       ,[2, 1, 8, 0, 1, 0]      # given a basic feasible solution.
                        ,[5, 2, 1, 0, 0, 1]])
b = np.array([20, 20, 20])

basis = np.array([0,1,2])                      # Starting with vertex associated with the initial basic feasible solution.
                                               # Every vertex in R^n x have n lin. indep. constraints active at x.
                                               # For every point in our polyhedron m constraints in the Ax = b active. 
                                               # We need n-m more so we set n-m variables to zero and call them non-basic to have a vertex.
non_basis = np.array([3,4,5])                  # The remaining m variables are called basic variables and their value is determined by the non-basic
                                               # variables (which we set to 0)

basis_matrix = np.array(constraints[:,basis])  # Matrix with the columns with basis indices i.e A_b(1):A_b(2):...:A_b(m)
cost_change = np.zeros(4, ) - 1               # Initialize with a negative value so the loop starts

while cost_change.min() < 0:

    inv_basis = np.linalg.inv(basis_matrix)                 # Calculating the inverse of the basis matrix, A_B^-1 in the notes.
    x_b = inv_basis @ b                                     # Values of the basic variables as noted above. the remaining variables are 0
    
    # given a non-basic variable x_j, define the j'th basic direction d as d_b := -inv_basis*A_j , d_j = 1 , d_i = 0 when i != j for i in non_basis

    cost_change = cost - cost[basis]@inv_basis@constraints  # Then the j'th component of the cost_change vector gives you the change in cost
                                                            # in the j'th basic direction.

    if cost_change.min() >= -epsilon:   # If cost_change in all the basic directions are positive STOP, you are at an optimal solution.
        print("optimal basis :", basis)
        print("values of the basic variables:", x_b)
        print("optimum_cost:", np.dot(x_b, cost[basis]))
        break

    print("cost_change: ", cost_change)

    entering_var_pos = np.argmax(-cost_change[non_basis]) # Index of the entering non-basic variable in the array non_basis
                                                          # chosen by most negative change in cost

    j = non_basis[entering_var_pos]                       # Global index of the entering variable is denoted by j i.e x_j is entering the basis

    print("Entering variable index:", j)

    a_j = constraints[:,j]                                # j'th column of A

    u = inv_basis@a_j                   # Minus the value of the change in the basic variables when we move in the j'th basic direction i.e u = -d_b
                                        # If d_b >= 0 for every basic variable then basic variables will never hit a ">= 0" constraint
                                        # Thus, we can go in the j'th basic direction forever while making the cost lower

    if u.max() <= 0:                    # if all the values of u are negative, d_b values are positive ^
        print("Optimal cost is -inf")
    else:                               # Now we need to determine the basic variable that'll hit 0 first when we go in the j'th basic direction.
        valid_indices = np.where(u > epsilon)[0] # Get the indices of the basic variables i that have u_i > 0
        print("valid indices:", valid_indices)

        ratios = x_b[valid_indices] / u[valid_indices] # This ratio is the step size needed to hit 0 for each basic variable
        min_ratio_index = np.argmin(ratios)            # Get the index of the variable that'll hit 0 first
        step_size = ratios[min_ratio_index]            # Denoted θ^* in the notes, θ^* = max_θ:A(x + θd) /in P, the step size we're taking

        exiting_variable_ind_basis = valid_indices[min_ratio_index] #This is just index madness :( Denoted l in the notes.
        
        # Valid indices are the global indices of the variables with positive u value indexed by the order in basis. 
        # So this value above is the index of the variable leaving the basis.

        non_basis[entering_var_pos] = basis[exiting_variable_ind_basis] # Switching the non-basic variable in the non_basis with the basic variable.
        basis[exiting_variable_ind_basis] = j                           # Vice versa

        basis_matrix = np.array(constraints[:,basis])  #Matrix with the columns with basis indices
    print("x_b: ", x_b)


