#include "devoir_2.h"
#include "utils.h"
#include "model.h"
#include "utils_gmsh.h"
#include <math.h>
#include <cblas.h>
#include <gmshc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VERBOSE 1
#define PRECISION 10


/**
 * Dump mesh (nodes, boundary edges, elements) into a simple text format.
 * Nodes:    "Number of nodes N" followed by "i : x y"
 * Edges:    "Number of edges M" followed by "i : n1 n2"
 * Elements: "Number of elements K" followed by "i : n1 n2 n3 ..."
 */
static void writeMeshTxt(const char *filename, FE_Model *model) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: cannot open '%s' for writing mesh\n", filename);
        exit(EXIT_FAILURE);
    }

    // 1) Nodes
    size_t nnode = model->n_node;
    fprintf(file, "Number of nodes %zu\n", nnode);
    // coords is [2*n_node] array: (x0,y0,x1,y1,...)
    for (size_t i = 0; i < nnode; ++i) {
        double x = model->coords[2*i];
        double y = model->coords[2*i + 1];
        fprintf(file, "%6zu : %14.7e %14.7e\n", i, x, y);
    }

    // 2) Boundary edges
    size_t nedges = model->n_bd_edge;
    fprintf(file, "Number of edges %zu\n", nedges);
    // bd_edges is [4* n_bd_edge]: (n1,n2,tag,length) 1-based node indices
    for (size_t e = 0; e < nedges; ++e) {
        size_t n1 = model->bd_edges[4*e] - 1;
        size_t n2 = model->bd_edges[4*e + 1] - 1;
        fprintf(file, "%6zu : %6zu %6zu\n", e, n1, n2);
    }

    // 3) Elements (triangles or quads)
    size_t nelem = model->n_elem;
    size_t nloc  = model->n_local;
    fprintf(file, "Number of triangles %zu\n", nelem);
    // elem_nodes is [n_local * n_elem]: 1-based node indices
    for (size_t e = 0; e < nelem; ++e) {
        fprintf(file, "%6zu :", e);
        for (size_t j = 0; j < nloc; ++j) {
            size_t nid = model->elem_nodes[nloc*e + j] - 1;
            fprintf(file, " %6zu", nid);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void display_sol(FE_Model *model, double *sol) {
    int ierr, n_views, *views;
    double *bounds;
    add_gmsh_views(&views, &n_views, &bounds);

    double *data_forces = malloc(6 * model->n_bd_edge * sizeof(double));
    visualize_disp(model, sol, views[1], 0, &bounds[2]);
    visualize_stress(model, sol, views, 1, 0, data_forces, bounds);
    visualize_bd_forces(model, data_forces, views[0], 1, &bounds[0]);

    create_tensor_aliases(views);
    set_view_options(n_views, views, bounds);
    gmshFltkRun(&ierr);
    gmshFltkFinalize(&ierr);
    free(data_forces);
}

void display_info(FE_Model *model, int step, struct timespec ts[4]) {

    char *m_str[3] = {"Plane stress", "Plane strain", "Axisymmetric"};
    char *r_str[4] = {"No", "X", "Y", "RCMK"};

    if (step == 1) {
        printf(
            "\n===========  Linear elasticity simulation - FEM  ===========\n\n"
        );
        printf("%30s = %s\n", "Model", model->model_name);
        printf("%30s = %s\n", "Model type", m_str[model->m_type]);
        printf("%30s = %.3e\n", "Young's Modulus E", model->E);
        printf("%30s = %.3e\n", "Poisson ratio nu", model->nu);
        printf("%30s = %.3e\n\n", "Density rho", model->rho);
    } else if (step == 2) {
        char *e_str = (model->e_type == TRI) ? "Triangle" : "Quadrilateral";
        printf("%30s = %s\n", "Element type", e_str);
        printf("%30s = %zu\n", "Number of elements", model->n_elem);
        printf("%30s = %zu\n", "Number of nodes", model->n_node);
        printf("%30s = %s\n", "Renumbering", r_str[model->renum]);
        printf("%30s = %zu\n\n", "Matrix bandwidth", 2 * model->node_band + 1);
    }
}


void write_csr_symmetric_market(const char *filename, const CSRMatrix *A) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    int n = A->n;
    // 1) compter les nnz_lower = nombre d'éléments avec i >= j
    int nnz_lower = 0;
    for (int i = 0; i < n; ++i) {
        for (int idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; ++idx) {
            int j = A->col_idx[idx];
            if (i >= j) ++nnz_lower;
        }
    }

    // 2) écrire l'en-tête MatrixMarket
    fprintf(f, "%%%%MatrixMarket matrix coordinate real symmetric\n");
    fprintf(f, "%d %d %d\n", n, n, nnz_lower);

    // 3) écrire les lignes i>=j
    for (int i = 0; i < n; ++i) {
        for (int idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; ++idx) {
            int j = A->col_idx[idx];
            double v = A->data[idx];
            if (i >= j) {
                // +1 pour passer au format 1-based
                fprintf(f, "%d %d %.15e\n", i+1, j+1, v);
            }
        }
    }

    fclose(f);
}


int main(int argc, char *argv[]) {
    int ierr;
    double mesh_size_ratio;
    if ((argc < 3) || (sscanf(argv[2], "%lf", &mesh_size_ratio)) != 1) {
        printf("Usage: \n./deformation <model> <mesh_size_ratio>\n");
        printf("model: one of the model implemented in models/\n");
        printf("mesh_size_ratio: mesh size factor\n");
        return -1;
    }

    // Simulation parameters
    const ElementType e_type = TRI;
    const Renumbering renum = RENUM_NO;  // let gmsh do the RCMK renumbering

    FE_Model *model = create_FE_Model(argv[1], e_type, renum);
    display_info(model, 1, NULL);
    gmshInitialize(argc, argv, 0, 0, &ierr);
    gmshOptionSetNumber("General.Verbosity", 2, &ierr);
    model->mesh_model(mesh_size_ratio, e_type);

    load_mesh(model);
    renumber_nodes(model);
    display_info(model, 2, NULL);
    assemble_system(model);
    double *rhs = (double *)calloc(2 * model->n_node, sizeof(double));
    double *sol = (double *)calloc(2 * model->n_node, sizeof(double));
    add_bulk_source(model, rhs);
    enforce_bd_conditions(model, rhs);

    
    SymBandMatrix *Kbd = model->K;
    SymBandMatrix *Mbd = model->M;
    CSRMatrix *Ksp = band_to_sym_csr(Kbd);
    CSRMatrix *Msp = band_to_sym_csr(Mbd);
    double eps = 1e-15;


    //** Added part **//

    /*Write the mesh to a file for plotting : Not usful anymore
    char mesh_file[100];
    sprintf(mesh_file, "./plot/%s.txt", model->model_name);
    writeMeshTxt(mesh_file, model);
    printf("Mesh written to %s\n", mesh_file);*/

    // First read the other arguments from the command line
    double T, dt; // Final time and time step
    int I_node; // Node of interest for time_file 
    char initial_file[200], final_file[200], time_file[200]; // File names for initial conditions, final values of all nodes and values of the node of interest over time

    if (argc < 8) {
        printf("Usage: \n./deformation <model> <mesh_size_ratio> <T> <dt> <initial_file> <final_file> <time_file> <I> [<compute_energy> <energy_file> <save_matrix> <animation>]\n");
        printf("compute_energy: -e to indicate whether to compute system energy over time\n");
        printf("energy_file: file to store energy values over time (required if compute_energy=1)\n");
        printf("save_matrix: -m to indicate whether to save the stiffness and mass matrices in .mtx format\n");
        printf("animation: -a to indicate whether to save the animation files\n");
        return -1;
    }

    // Read the parameters from the command line
    sscanf(argv[3], "%lf", &T);
    sscanf(argv[4], "%lf", &dt);
    sscanf(argv[5], "%s", initial_file);
    sscanf(argv[6], "%s", final_file);
    sscanf(argv[7], "%s", time_file);
    sscanf(argv[8], "%d", &I_node);

    int compute_energy = 0;
    char energy_file[100];
    int save_matrix = 0;
    int animation = 0;
    if (argc > 8) {
        for (int i = 9; i < argc; ++i) {
            if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
                compute_energy = 1;
                strncpy(energy_file, argv[i + 1], sizeof(energy_file) - 1);
                energy_file[sizeof(energy_file) - 1] = '\0'; // Ensure null-termination
                i++; // Skip next arg since it's the filename
            } else if (strcmp(argv[i], "-m") == 0) {
                save_matrix = 1;
            } else if (strcmp(argv[i], "-a") == 0) {
                animation = 1;
            }
            else {
                printf("Unknown optional argument: %s\n", argv[i]);
                exit(EXIT_FAILURE);
            }
        }
    }

    if(save_matrix) {
        write_csr_symmetric_market("K.mtx", Ksp);
        write_csr_symmetric_market("M.mtx", Msp);
    }

    // We read the initial conditions from the file
    double *u = (double *)malloc(2 * model->n_node * sizeof(double));
    double *v = (double *)malloc(2 * model->n_node * sizeof(double));
    read_initial_conditions(initial_file, u, v, model->n_node);
   
    // Choose the Newmark parameters so that the method is unconditionally stable and symplectic
    double beta = 0.25; // Newmark parameter
    double gamma = 0.5; // Newmark parameter

    double tau = model->L_ref * sqrt(model->rho/model->E); // Characteristic time of the system

    // Compute Invsp = M+beta*dt^2*K which we are going to use in the Newmark method
    // First compute it in band form then convert to CSR

    int k = Mbd->k > Kbd->k ? Mbd->k : Kbd->k;
    SymBandMatrix *Inv = allocate_sym_band_matrix(Msp->n, k);
    // Add M to Inv
    for (int i = 0; i < Mbd->n; i++) {
        int limit = 0 < Mbd->k ? Mbd->k : 0;
        for (int j = 0; j <= limit; j++) {
            Inv->a[i][i-j] = Mbd->a[i][i-j];
        }
    }
    // Add beta*dt^2*K to Inv
    for (int i = 0; i < Kbd->n; i++) {
        int limit = 0 < Kbd->k ? Kbd->k : 0;
        for (int j = 0; j <= limit; j++) {
            Inv->a[i][i-j] += beta * dt * dt * Kbd->a[i][i-j];
        }
    }
    // Convert to CSR
    CSRMatrix *Invsp = band_to_sym_csr(Inv);
    free_sym_band_matrix(Inv); // Free the band matrix
    // Iterate over time and fill the files with the results
    newmark(u, v, dt, T, model->n_node, Ksp, Msp, Invsp, beta, gamma, final_file, time_file, I_node, eps, compute_energy, energy_file, model->E, model->L_ref, tau, animation);


    free_csr(Invsp);
    free(v);
    //** End of added part **//

    // Display the final solution
    display_sol(model, u);
    free(u);

    // Free stuff
    free_csr(Ksp);
    free_csr(Msp);
    gmshFinalize(&ierr);
    free(sol);
    free(rhs);
    free_FE_Model(model);
    return 0;
}
