#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <numeric>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace std;
using Clock = chrono::steady_clock;

// --- Estructuras ---
struct Route {
    vector<int> nodes;
    double cost;
    int load;
    vector<double> prefix_cost;
    int vehicle_id;
};
struct Solution {
    vector<Route> routes;
    double total_cost;
};

// --- Parámetros de configuración ---
static const int OPENMP_THRESHOLD = 5;   // mínimo rutas para paralelizar
static const int TOP_K_INSERT = 10;      // candidatos para inserción

// --- Funciones Auxiliares ---

// Actualiza coste y prefix_cost de ruta
void update_route(Route &r, const vector<vector<double>>& D) {
    int n = r.nodes.size();
    r.prefix_cost.resize(n+1);
    double acc = 0.0; int prev = 0;
    r.prefix_cost[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += D[prev][r.nodes[i]];
        prev = r.nodes[i];
        r.prefix_cost[i+1] = acc;
    }
    acc += D[prev][0];
    r.cost = acc;
}

// Comprueba factibilidad según capacidad del vehículo
inline bool feasible_load(const Route &r,
    const vector<int>& dem, const vector<int>& caps)
{
    return r.load <= caps[r.vehicle_id];
}

// Clarke-Wright Savings con asignación heterogénea
Solution initial_solution_hetero_savings(
    const vector<vector<double>>& D,
    const vector<int>& dem,
    const vector<int>& caps)
{
    int m = dem.size() - 1;
    int K = caps.size();
    // Lista de rutas, inicial vacías
    vector<Route> R;
    R.reserve(K);
    for (int i = 0; i < K; ++i)
        R.push_back(Route{{},0.0,0,{},i});

    // Savings: (i,j,savings)
    struct Sav {int i,j; double s;};
    vector<Sav> S;
    for (int i = 1; i <= m; ++i) for (int j = i+1; j <= m; ++j) {
        double sij = D[0][i] + D[0][j] - D[i][j];
        S.push_back({i,j,sij});
    }
    sort(S.begin(), S.end(), [](auto &a, auto &b){ return a.s > b.s; });

    // Inicial: cada cliente como ruta temporal
    vector<Route> temp;
    for (int i = 1; i <= m; ++i) {
        temp.push_back(Route{{i}, D[0][i]*2, dem[i], {0.0, D[0][i]}, -1});
    }

    // Fusionar según savings respetando capacidades
    for (auto &sav : S) {
        int ri=-1, rj=-1;
        for (int k = 0; k < temp.size(); ++k) {
            auto &nodes = temp[k].nodes;
            if (nodes.front()==sav.i) ri = k;
            if (nodes.back()==sav.j)  rj = k;
        }
        if (ri>=0 && rj>=0 && ri!=rj) {
            int combined_load = temp[ri].load + temp[rj].load;
            // asignar a primer vehículo que lo acepte
            for (int v = 0; v < K; ++v) {
                if (combined_load <= caps[v]) {
                    temp[ri].nodes.insert(
                        temp[ri].nodes.end(),
                        temp[rj].nodes.begin(), temp[rj].nodes.end());
                    temp[ri].load = combined_load;
                    update_route(temp[ri], D);
                    temp.erase(temp.begin()+rj);
                    temp[ri].vehicle_id = v;
                    break;
                }
            }
        }
    }

    // Insert restantes con top-K best insertion
    vector<bool> used(m+1,false);
    for (auto &r: temp) for (int c: r.nodes) used[c]=true;
    vector<int> leftovers;
    for (int i = 1; i <= m; ++i) if (!used[i]) leftovers.push_back(i);

    for (int c : leftovers) {
        struct Ins{double cost; int idx; int pos; int vid;};
        vector<Ins> cand;
        for (int v = 0; v < K; ++v) {
            if (dem[c] > caps[v]) continue;
            auto &route = temp[v];
            int n = route.nodes.size();
            for (int p = 0; p <= n; ++p) {
                int prev = (p? route.nodes[p-1] : 0);
                int next = (p<n? route.nodes[p] : 0);
                double delta = D[prev][c] + D[c][next] - D[prev][next];
                cand.push_back({delta, v, p, v});
            }
        }
        sort(cand.begin(), cand.end(), [](auto &a, auto &b){ return a.cost < b.cost; });
        if (!cand.empty()) {
            auto &best = cand.front();
            auto &r = temp[best.idx];
            r.nodes.insert(r.nodes.begin()+best.pos, c);
            r.load += dem[c];
            r.vehicle_id = best.vid;
            update_route(r, D);
        }
    }

    // Asignar soluciones a R (heterogéneo)
    for (int i = 0; i < temp.size() && i < K; ++i)
        R[i] = temp[i];

    double tot = 0.0;
    for (auto &r: R) { update_route(r, D); tot += r.cost; }
    return {R, tot};
}

// --- Operadores de core ---
enum OpType { SWAP=0, RELOC=1, TWOOPT=2, OP_COUNT=3 };

bool op_swap(Route &A, Route &B) {
    if (A.nodes.empty()||B.nodes.empty()) return false;
    static thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> ia(0,A.nodes.size()-1), ib(0,B.nodes.size()-1);
    swap(A.nodes[ia(gen)], B.nodes[ib(gen)]);
    return true;
}

bool op_relocate(Route &A, Route &B, const vector<int>& dem) {
    if (A.nodes.empty()) return false;
    static thread_local mt19937 gen(random_device{}());
    int pa = uniform_int_distribution<>(0,A.nodes.size()-1)(gen);
    int c = A.nodes[pa];
    A.nodes.erase(A.nodes.begin()+pa); A.load -= dem[c];
    int pb = uniform_int_distribution<>(0,B.nodes.size())(gen);
    B.nodes.insert(B.nodes.begin()+pb, c); B.load += dem[c];
    return true;
}

bool op_twoopt(Route &A) {
    int n = A.nodes.size(); if (n < 4) return false;
    static thread_local mt19937 gen(random_device{}());
    int i = uniform_int_distribution<>(0,n-2)(gen);
    int j = uniform_int_distribution<>(i+1,n-1)(gen);
    reverse(A.nodes.begin()+i, A.nodes.begin()+j);
    return true;
}

// --- VND con pesos y actualización ---
void VND(Route &r,
         const vector<vector<double>>& D,
         const vector<int>& dem,
         const vector<int>& caps,
         vector<double>& op_weights)
{
    vector<OpType> ops = {SWAP,RELOC,TWOOPT};
    int k = 0;
    while (k < OP_COUNT) {
        Route r2 = r;
        bool ok = false;
        switch (ops[k]) {
            case SWAP:    ok = op_swap(r2,r2); break;
            case RELOC:   ok = op_relocate(r2,r2,dem); break;
            case TWOOPT:  ok = op_twoopt(r2);        break;
        }
        if (ok) {
            update_route(r2, D);
            if (feasible_load(r2, dem, caps) && r2.cost < r.cost) {
                r = move(r2);
                op_weights[ops[k]] *= 1.2;
                k = 0; continue;
            }
        }
        ++k;
    }
}

// --- VNS mejorado heterogéneo ---
Solution vns_hetero_improved(
    const vector<vector<double>>& D,
    const vector<int>& dem,
    const vector<int>& caps,
    int max_iter = 2000,
    double time_limit = 10.0)
{
    auto start = Clock::now();
    int K = caps.size();
    // Inicial
    Solution best = initial_solution_hetero_savings(D,dem,caps);
    Solution curr = best;
    vector<double> op_weights(OP_COUNT,1.0);
    mt19937 gen(random_device{}()); uniform_real_distribution<> prob(0,1);
    int stagn=0;
    double shake_i = 2.0;

    for (int it=0; it<max_iter && stagn < max_iter/5; ++it) {
        if (chrono::duration<double>(Clock::now()-start).count()>time_limit) break;
        int shakes = max(1,int(shake_i));
        Solution trial = curr;
        // Shaking adaptativo
        for (int s=0; s<shakes; ++s) {
            double sumw = accumulate(op_weights.begin(), op_weights.end(),0.0);
            double r = prob(gen)*sumw, acc=0;
            OpType chosen = SWAP;
            for (int o=0;o<OP_COUNT;++o) {
                acc += op_weights[o]; if (r<=acc) { chosen=OpType(o); break; }
            }
            int i = uniform_int_distribution<>(0,K-1)(gen);
            int j = uniform_int_distribution<>(0,K-1)(gen);
            switch (chosen) {
                case SWAP:   op_swap(trial.routes[i],trial.routes[j]); break;
                case RELOC:  op_relocate(trial.routes[i],trial.routes[j],dem); break;
                case TWOOPT: op_twoopt(trial.routes[i]); break;
            }
        }
        // Local search: decidir usar o no OpenMP
        if (K >= OPENMP_THRESHOLD) {
        #pragma omp parallel for schedule(dynamic)
            for (int i=0;i<K;++i)
                VND(trial.routes[i],D,dem,caps,op_weights);
        } else {
            for (int i=0;i<K;++i)
                VND(trial.routes[i],D,dem,caps,op_weights);
        }
        // Recalcular coste total
        trial.total_cost = 0;
        for (auto &r: trial.routes) { update_route(r,D); trial.total_cost += r.cost; }
        // Evaluación
        if (trial.total_cost < best.total_cost) {
            best = curr = trial;
            stagn = 0;
            for (auto &w: op_weights) w = max(1.0, w*0.9);
            shake_i = 2.0;
        } else {
            curr = trial;
            stagn++;
            shake_i = min(10.0, shake_i*1.05);
        }
    }
    return best;
}

// --- Interfaz Python ---
PYBIND11_MODULE(vns_solver, m) {
    py::class_<Route>(m, "Route")
        .def_readonly("nodes", &Route::nodes)
        .def_readonly("cost",  &Route::cost)
        .def_readonly("load",  &Route::load)
        .def_readonly("vehicle_id", &Route::vehicle_id);
    py::class_<Solution>(m, "Solution")
        .def_readonly("routes",     &Solution::routes)
        .def_readonly("total_cost", &Solution::total_cost);
    m.def("vns_hetero_improved", &vns_hetero_improved,
          py::arg("distance_matrix"),
          py::arg("demands"),
          py::arg("capacities"),
          py::arg("max_iterations") = 2000,
          py::arg("time_limit") = 10.0);
}
