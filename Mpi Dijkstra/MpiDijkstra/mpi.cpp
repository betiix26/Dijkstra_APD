
/*
 * Input:   n -> nr de varfuri
 *          mat -> matricea, unde mat[i][j] este lungimea de la varful i la j 
 *
 * Output:  lungimea celui mai scurt drum de la varful 0 la varful v (Dijkstra's shortest path algorithm)
 *       
 * Rulare: mpiexec -n 6 MpiDijkstra.exe 
 * 
 * Algoritm: Matricea este impartita pe coloane, astfel incat fiecare proces sa primeasca n / p coloane.
 * In fiecare iteratie, fiecare proces isi gaseste varful local cu cea mai scurta distanta de la varful sursa 0. 
 * Se calculeaza un varf minim global u al celor mai scurte distante gasite si apoi 
 * fiecare proces isi actualizeaza matricea de distante locale daca exista o cale mai scurta care trece prin u.
 * 
 *          1. n trebuie sa se imparta fix la p (nr de procese)
 *          2. Valorile muchiilor trebuie sa fie pozitive
 *          3. Daca nu exista muchie intre 2 noduri, atunci valoarea muchiei este constanta INFINITY
 *          4. Costul unui drum de la un nod la el insusi este 0
 *          5. Matricea de adiacenta este stocata ca o matrice unidimensionala si indicele sunt calculate 
 *            folosind A[n * i + j] pentru a obtine A[i][j] in cazul bidimensional
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define INFINITY 1000000

int Read_n(int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(int loc_mat[], int n, int loc_n, MPI_Datatype blk_col_mpi_t,
    int my_rank, MPI_Comm comm);
void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
    int my_rank, int loc_n);
void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
    MPI_Comm comm);
int Find_min_dist(int loc_dist[], int loc_known[], int loc_n);
void Print_matrix(int global_mat[], int rows, int cols);
void Print_dists(int global_dist[], int n);
void Print_paths(int global_pred[], int n);

int main(int argc, char** argv) {
    int* loc_mat, * loc_dist, * loc_pred, * global_dist = NULL, * global_pred = NULL;
    int my_rank, p, loc_n, n;
    MPI_Comm comm;
    MPI_Datatype blk_col_mpi_t;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &p);
    n = Read_n(my_rank, comm);
    loc_n = n / p;
    loc_mat = (int*)malloc(n * loc_n * sizeof(int));
    loc_dist = (int*)malloc(loc_n * sizeof(int));
    loc_pred = (int*)malloc(loc_n * sizeof(int));
    blk_col_mpi_t = Build_blk_col_type(n, loc_n);

    if (my_rank == 0) {
        global_dist = (int*)malloc(n * sizeof(int));
        global_pred = (int*)malloc(n * sizeof(int));
    }
    Read_matrix(loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm);
    Dijkstra(loc_mat, loc_dist, loc_pred, loc_n, n, comm);

    //colectam rezutatele folosind gather
    MPI_Gather(loc_dist, loc_n, MPI_INT, global_dist, loc_n, MPI_INT, 0, comm);
    MPI_Gather(loc_pred, loc_n, MPI_INT, global_pred, loc_n, MPI_INT, 0, comm);

    // afiseaza rezultatele
    if (my_rank == 0) {
        Print_dists(global_dist, n);
        Print_paths(global_pred, n);
        free(global_dist);
        free(global_pred);
    }
    free(loc_mat);
    free(loc_pred);
    free(loc_dist);
    MPI_Type_free(&blk_col_mpi_t);
    MPI_Finalize();
    return 0;
}

int Read_n(int my_rank, MPI_Comm comm) { //citeste nr de randuri din matrice pe procesul 0 
                                        //transmite valoarea catre celelalte procese
//my_rank -> rangul procesului de apelare
//comm -> comunicatorul care contine toate procesele de apelare

    int n; 

    if (my_rank == 0)       
        scanf("%d", &n);    

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    return n; //n -> nr de randuri din matrice
}


MPI_Datatype Build_blk_col_type(int n, int loc_n) { //coloana intreaga a matricei
    //n -> nr de randuri din matrice si coloana
    //loc_n = n / p -> nr cols in coloana
    MPI_Aint lb, extent;
    MPI_Datatype block_mpi_t;
    MPI_Datatype first_bc_mpi_t;
    MPI_Datatype blk_col_mpi_t; //bloc de coloana

    MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
    MPI_Type_get_extent(block_mpi_t, &lb, &extent);

    /* MPI_Type_vector(numblocks, elts_per_block, stride, oldtype, *newtype) */
    MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);

    // apel necesar pentru a obtine masura corecta a noului tip de date 
    MPI_Type_create_resized(first_bc_mpi_t, lb, extent, &blk_col_mpi_t);

    MPI_Type_commit(&blk_col_mpi_t);

    MPI_Type_free(&block_mpi_t);
    MPI_Type_free(&first_bc_mpi_t);

    return blk_col_mpi_t;
}

void Read_matrix(int loc_mat[], int n, int loc_n,       //citim intr-o matrice nxn de int pe procesul 0 
    MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm) { //si o distribuim intre procese, a.i. 
    int* mat = NULL, i, j;                                     //fiecare proces sa primeasca o coloana intreaga cu n randuri si n/p coloane

    //n -> nr de coloane/linii in matrice si in submatrice
    //loc_n -> nr de coloane in submatrice
    //blk_col_mpi_t -> MPI_Datatype folosit pe procesul 0
    //my_rank -> rangul apelantului în comm
    //comm -> comunicatorul care contine toate procesele

    //loc_mat -> submatricea procesului de apelare (trebuie sa fie alocate de apelant)

    if (my_rank == 0) {
        mat = (int*) malloc(n * n * sizeof(int));
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                scanf("%d", &mat[i * n + j]);
    }

    MPI_Scatter(mat, 1, blk_col_mpi_t, loc_mat, n * loc_n, MPI_INT, 0, comm);

    if (my_rank == 0) free(mat);
}

void Dijkstra_Init(int loc_mat[], int loc_pred[], int loc_dist[], int loc_known[],
    int my_rank, int loc_n) {  //initializam toate matricele a.i. sa rulam cel mai mic drum
    //loc_n -> nr local de varfuri
    //my_rank -> rangul procesului
    //loc_mat -> matrice locala care contine costurile muchiilor intre vârfuri
    //loc_dist -> loc_dist[v] = cea mai scurta distanta de la sursa la fiecare varf v
    //loc_pred -> loc_pred[v] = predecesorul lui v pe cea mai scurta cale de la sursa la v
    //loc_known -> loc_known[v] = 1 daca varful a fost vizitat, 0 altfel
    int loc_v; 

    if (my_rank == 0)
        loc_known[0] = 1;
    else
        loc_known[0] = 0;

    for (loc_v = 1; loc_v < loc_n; loc_v++)
        loc_known[loc_v] = 0;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
        loc_pred[loc_v] = 0;
    }
}

void Dijkstra(int loc_mat[], int loc_dist[], int loc_pred[], int loc_n, int n,
    MPI_Comm comm) { //calculeaza toate caile cele mai scurte de la varful sursa 0 la toate varfurile v
//loc_mat -> matrice locala care contine costurile muchiilor intre varfuri
//loc_n -> nr local de varfuri
//n -> nr total de varfuri (la nivel global)
//comm -> comunicatorul
    int i, loc_v, loc_u, glbl_u, new_dist, my_rank, dist_glbl_u;
    int* loc_known;
    int my_min[2];
    int glbl_min[2];
//loc_dist -> loc_dist[v] = cea mai scurta distanta de la sursa la fiecare varf v
//loc_pred -> loc_pred[v] = predecesorul lui v pe cea mai scurta cale de la sursa la v
    MPI_Comm_rank(comm, &my_rank);
    loc_known = (int*)malloc(loc_n * sizeof(int));

    Dijkstra_Init(loc_mat, loc_pred, loc_dist, loc_known, my_rank, loc_n);

    //rulam bucla de n - 1 ori pentru ca stim deja cea mai scurta cale catre varful global 0 din varful global 0 
    for (i = 0; i < n - 1; i++) {
        loc_u = Find_min_dist(loc_dist, loc_known, loc_n);

        if (loc_u != -1) {
            my_min[0] = loc_dist[loc_u];
            my_min[1] = loc_u + my_rank * loc_n;
        }
        else {
            my_min[0] = INFINITY;
            my_min[1] = -1;
        }

        //obtinem distanta minima gasita de procese 
        //si stocam distanta si varful global in glbl_min
   
        MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);

        dist_glbl_u = glbl_min[0];
        glbl_u = glbl_min[1];

        //ne asiguram ca loc_known nu este accesat cu -1 
        if (glbl_u == -1)
            break;

        //verificam daca global_u apartine procesului 
        //si daca da, atunci actualizam loc_known
        if ((glbl_u / loc_n) == my_rank) {
            loc_u = glbl_u % loc_n;
            loc_known[loc_u] = 1;
        }

        /*
            Pentru fiecare varf local (global_vertex = loc_v + my_rank * loc_n)
          actualizam distantele de la varful sursa 0 la loc_v. Daca varful
          este nemarcat, atunci verificam daca distanta de la sursa la global_u + 
          distanta de la global_u la v local < decat distanta de la sursa la local_v*/
        for (loc_v = 0; loc_v < loc_n; loc_v++) {
            if (!loc_known[loc_v]) {
                new_dist = dist_glbl_u + loc_mat[glbl_u * loc_n + loc_v];
                if (new_dist < loc_dist[loc_v]) {
                    loc_dist[loc_v] = new_dist;
                    loc_pred[loc_v] = glbl_u;
                }
            }
        }
    }
    free(loc_known);
}


int Find_min_dist(int loc_dist[], int loc_known[], int loc_n) { //cauta distanta locala minima de la sursa la varfurile atribuite procesului care apeleaza metoda
//loc_dist -> matrice cu distante de la sursa 0
//loc_known -> matrice cu valorile 1 daca varful a fost vizitat, 0 daca nu
//loc_n -> nr local de varfuri
    int loc_u = -1; //nu se mai foloseste daca functia returneaza
    int loc_v; 
    int shortest_dist = INFINITY;

    for (loc_v = 0; loc_v < loc_n; loc_v++) {
        if (!loc_known[loc_v]) {
            if (loc_dist[loc_v] < shortest_dist) {
                shortest_dist = loc_dist[loc_v];
                loc_u = loc_v;
            }
        }
    }
    return loc_u;  
    //loc_u -> varful cu cea mai mica valoare in loc_dist, 
    //-1 daca toate varfurile sunt deja cunoscute
}

void Print_matrix(int mat[], int rows, int cols) { //afiseaza continutul matricei
    //mat -> matricea ; rows -> linii ; cols -> coloane
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++)
            if (mat[i * cols + j] == INFINITY)
                printf("i ");
            else
                printf("%d ", mat[i * cols + j]);
        printf("\n");
    }

    printf("\n");
}

void Print_dists(int global_dist[], int n) { //afiseaza lungimea celei mai scurte cai de la 0 la fiecare varf
    int v;
    //n -> nr de varfuri
    //dist -> distantele de la 0 la fiecare varf v
    //dist[v] -> lungimea celui mai scurt drum de la 0 -> v
    printf("  v    dist 0->v\n");
    printf("----   ---------\n");

    for (v = 1; v < n; v++) {
        if (global_dist[v] == INFINITY) {
            printf("%3d       %5s\n", v, "inf");
        }
        else
            printf("%3d       %4d\n", v, global_dist[v]);
    }
    printf("\n");
}

void Print_paths(int global_pred[], int n) { //afiseaza cel mai scurt drum de la 0 la fiecare varf
    int v, w, * path, count, i;
    //n -> nr de varfuri
    //pred -> lista de predecesori 
    //pred[v] = u daca u il precede pe v pe cel mai mic drum de la 0 -> v
    path = (int*)malloc(n * sizeof(int));

    printf("  v     Path 0->v\n");
    printf("----    ---------\n");
    for (v = 1; v < n; v++) {
        printf("%3d:    ", v);
        count = 0;
        w = v;
        while (w != 0) {
            path[count] = w;
            count++;
            w = global_pred[w];
        }
        printf("0 ");
        for (i = count - 1; i >= 0; i--)
            printf("%d ", path[i]);
        printf("\n");
    }

    free(path);
}
