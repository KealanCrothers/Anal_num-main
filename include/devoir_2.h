#ifndef DEVOIR_2_H
#define DEVOIR_2_H

#include "model.h"
#include <stdio.h>
void Matvec(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *v,
    double *Av
);

int CG(
    int n,
    const int *rows_idx,
    const int *cols,
    const double *A,
    const double *b,
    double *x,
    double eps
);

int csr_sym();

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
);

void read_initial_conditions(const char *filename, double *u, double *v, int n_node);

#endif
