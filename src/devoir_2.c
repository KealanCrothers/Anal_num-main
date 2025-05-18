#include "devoir_2.h"
#include "model.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gmshc.h>

#define SYM 1 
int csr_sym() { return SYM; }

void Matvec(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *v,
    double *Av
) {

    cblas_dscal(n, 0, Av, 1);
    for (int i = 0; i < n; i++) {
        for (int j = rows_idx[i]; j < rows_idx[i + 1] - 1; j++) {
            Av[i] += A[j] * v[cols[j]];
            Av[cols[j]] += A[j] * v[i];
        }
        Av[i] += A[rows_idx[i + 1] - 1] * v[i];
    }
}

int CG(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps
) {
    int idx = 0;
    double alpha, beta;
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    double *r = (double *)malloc(n * sizeof(double));

    cblas_dscal(n, 0, x, 1);
    cblas_dcopy(n, b, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    double r0 = cblas_dnrm2(n, b, 1);
    double r_norm2 = cblas_ddot(n, r, 1, r, 1);
    // printf("r0 = %9.3le\n", r0);

    while (sqrt(r_norm2) / r0 > eps) {
        // if (idx %100 == 0) printf("idx : %4d\n", idx);
        Matvec(n, rows_idx, cols, A, p, Ap);
        alpha = r_norm2 / cblas_ddot(n, p, 1, Ap, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, Ap, 1, r, 1);
        beta = 1 / r_norm2;
        r_norm2 = cblas_ddot(n, r, 1, r, 1);
        beta *= r_norm2;
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);
        idx++;
        // printf("it : %3d  -> res = %9.3le\n", idx, sqrt(r_norm2) / r0);
    }

    // printf("cg it : %d\n", idx);
    free(p);
    free(Ap);
    free(r);
    return idx;
}


void read_initial_conditions(const char *filename, double *u, double *v, int n_node) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening initial conditions file");
        exit(EXIT_FAILURE);
    }
    // Each line contains ux_i, uy_i, ux_dot_i, uy_dot_i for each node i
    for (int i = 0; i < n_node; i++) {
        if (fscanf(file, "%le %le %le %le", &u[2 * i], &u[2 * i + 1], &v[2 * i], &v[2 * i + 1]) != 4) {
            fprintf(stderr, "Error reading initial conditions for node %d\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void write_time(FILE * file, double t, double *u, double *v, int I_node) {
   fprintf(file, "%.151e %.151e %.151e %.151e %.151e\n", t, u[2 * I_node], u[2 * I_node + 1], v[2 * I_node], v[2 * I_node + 1]);
}

void write_energy(FILE * file, double t, double *u, double *v, CSRMatrix *K, CSRMatrix *M, int n_node, double *temp, double E, double Lref, double tau) {
    // Compute potential energy defined as u^T K u /2 
    Matvec(K->n, K->row_ptr, K->col_idx, K->data, u, temp);
    double potential_energy = cblas_ddot(n_node, u, 1, temp, 1) / 2.0;
    // Compute kinetic energy defined as v^T M v /2
    Matvec(M->n, M->row_ptr, M->col_idx, M->data, v, temp);
    double kinetic_energy = cblas_ddot(n_node, v, 1, temp, 1) / 2.0;

    // Scale the energies to get the energy in Joules
    potential_energy *= Lref * Lref * E;
    kinetic_energy *= Lref * Lref * E;

    // Scale the time to get the time in seconds
    t *= tau;

    // Write the energies to the file
    fprintf(file, "%.151e %.151e %.151e\n", t, potential_energy, kinetic_energy);
}

void writeDispTxt(const char *filename, size_t n_node, double *u) {
    // Used for the animation
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: cannot open '%s' for writing solution", filename);
        exit(EXIT_FAILURE);
    }
    int nfields = 2; // ux, uy
    fprintf(file, "Size %zu,%d\n", n_node, nfields);
    for (size_t i = 0; i < n_node; ++i) {
        double ux = u[2*i];
        double uy = u[2*i + 1];
        fprintf(file, "%.18le,%.18le\n", ux, uy);
    }
    fclose(file);
}

void newmark(
    double *u,
    double *v,
    double dt,
    double T,
    int n_node,
    CSRMatrix *Ksp,
    CSRMatrix *Msp,
    CSRMatrix *Invsp,
    double beta,
    double gamma,
    const char *final_file,
    const char *time_file,
    int I_node,
    double eps,
    int compute_energy,
    const char *energy_file,
    double E,
    double Lref,
    double tau,
    int anim
) {
    // Open the output files
    FILE *final_file_ptr = fopen(final_file, "w");
    FILE *time_file_ptr = fopen(time_file, "w");
    
    // If compute_energy is true, open the energy file
    FILE *energy_file_ptr;
    if(compute_energy) 
        energy_file_ptr = fopen(energy_file, "w");

    if (final_file_ptr == NULL || time_file_ptr == NULL) {
        perror("Error opening output files");
        exit(EXIT_FAILURE);
    }

    int n = n_node*2; // Number of degrees of freedom

    // Allocate memory for the right-hand side vector and the solution vector
    double *temp = (double *)malloc(n * sizeof(double));
    double *temp2 = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double)); // p_n is Mv_n

    int nb_time_steps = (int)(round(T / dt));

    // Compute p_0 defined as Mv_0
    Matvec(Msp->n, Msp->row_ptr, Msp->col_idx, Msp->data, v, p);


    // Start the iterations
    for(int step = 0; step <= nb_time_steps; step++) {
        if(anim)
            printf("step : %d\n", step);

        double t = step * dt;
        // Write the current u and v to the file
        write_time(time_file_ptr, t, u, v, I_node);
        if (compute_energy) {
            write_energy(energy_file_ptr, t, u, v, Ksp, Msp, n, temp, E, Lref, tau);
        }

        if(anim){
            // Write the current u into the animation file
            char animation_file[100];
            sprintf(animation_file, "./plot/animation/animation_%d.txt", step);
            writeDispTxt(animation_file, n_node, u);
        }

        if(step == nb_time_steps) break;

        // Update u : (Invsp)*u_n+1 = M*u_n - ((1-2*beta)*dt^2/2) * K*u_n) + dt*p_n
        Matvec(Msp->n, Msp->row_ptr, Msp->col_idx, Msp->data, u, temp);
        Matvec(Ksp->n, Ksp->row_ptr, Ksp->col_idx, Ksp->data, u, temp2);
        cblas_daxpy(n, -((1-2*beta)*dt*dt)/2.0, temp2, 1, temp, 1);
        cblas_daxpy(n, dt, p, 1, temp, 1);
        CG(Invsp->n, Invsp->row_ptr, Invsp->col_idx, Invsp->data, temp, temp2, eps);

        // Update p : p_n+1 = p_n - dt * K * ((1-gamma)*u_n + gamma*u_n+1)
        cblas_dscal(n, 1-gamma, u, 1);
        cblas_daxpy(n, gamma, temp2, 1, u, 1);
        Matvec(Ksp->n, Ksp->row_ptr, Ksp->col_idx, Ksp->data, u, temp);
        cblas_daxpy(n, -dt, temp, 1, p, 1);
        cblas_dcopy(n, temp2, 1, u, 1);

        // Update v : p_n+1 = Mv_n+1
        CG(Msp->n, Msp->row_ptr, Msp->col_idx, Msp->data, p, v, eps);
    }
    
    // Write the final u and v to the final file
    for(int i = 0; i < n_node; i++) fprintf(final_file_ptr, "%.151e %.151e %.151e %.151e\n", u[2 * i], u[2 * i + 1], v[2 * i], v[2 * i + 1]);

    // Close the output files
    fclose(final_file_ptr);
    fclose(time_file_ptr);
    fclose(energy_file_ptr);
    
    // Free the allocated memory
    free(temp);
    free(temp2);
    free(p);
}






    


    
