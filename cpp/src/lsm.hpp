#pragma once

#include <vector>

double lsm_price_from_paths(
    const double* paths,   // contiguous array (n_paths x (n_steps+1))
    int n_paths,
    int n_steps,
    double K,
    double r,
    double T,
    bool is_call,
    int degree
);
