#include "lsm.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

static inline double payoff(double S, double K, bool is_call) {
    return is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
}

double lsm_price_from_paths(
    const double* paths,
    int n_paths,
    int n_steps,
    double K,
    double r,
    double T,
    bool is_call,
    int degree
) {
    if (!paths) throw std::invalid_argument("paths pointer is null");
    if (n_paths <= 0) throw std::invalid_argument("n_paths must be positive");
    if (n_steps <= 0) throw std::invalid_argument("n_steps must be positive");
    if (T <= 0.0) throw std::invalid_argument("T must be positive");
    if (degree < 0) throw std::invalid_argument("degree must be non-negative");

    const int n_cols = n_steps + 1;
    const double dt = T / static_cast<double>(n_steps);
    const double disc = std::exp(-r * dt);

    // cashflow at maturity (undiscounted at maturity time)
    std::vector<double> cashflow(n_paths);
    for (int i = 0; i < n_paths; ++i) {
        const double ST = paths[i * n_cols + n_steps];
        cashflow[i] = payoff(ST, K, is_call);
    }

    // backward induction: t = n_steps-1 ... 1 (skip 0)
    for (int t = n_steps - 1; t >= 1; --t) {
        // discount cashflows one step back to time t for everyone
        for (int i = 0; i < n_paths; ++i) {
            cashflow[i] *= disc;
        }

        // collect ITM indices
        std::vector<int> itm_idx;
        itm_idx.reserve(n_paths);

        for (int i = 0; i < n_paths; ++i) {
            const double St = paths[i * n_cols + t];
            const double imm = payoff(St, K, is_call);
            if (imm > 0.0) itm_idx.push_back(i);
        }

        const int m = static_cast<int>(itm_idx.size());
        const int p = degree + 1;
        if (m < p || m == 0) {
            // not enough points to regress continuation
            continue;
        }

        // Build regression X (m x p) and Y (m)
        Eigen::MatrixXd X(m, p);
        Eigen::VectorXd Y(m);

        for (int row = 0; row < m; ++row) {
            const int i = itm_idx[row];
            const double St = paths[i * n_cols + t];

            // polynomial basis [1, S, S^2, ...]
            double powS = 1.0;
            for (int col = 0; col < p; ++col) {
                X(row, col) = powS;
                powS *= St;
            }
            Y(row) = cashflow[i];
        }

        // Least squares: beta = argmin ||X beta - Y||
        Eigen::VectorXd beta = X.colPivHouseholderQr().solve(Y);

        // Exercise decision on ITM paths
        for (int row = 0; row < m; ++row) {
            const int i = itm_idx[row];
            const double St = paths[i * n_cols + t];
            const double imm = payoff(St, K, is_call);

            // continuation = basis(St) dot beta
            double cont = 0.0;
            double powS = 1.0;
            for (int col = 0; col < p; ++col) {
                cont += beta(col) * powS;
                powS *= St;
            }

            if (imm > cont) {
                cashflow[i] = imm;  // at time t (already discounted to time t by earlier step)
            }
        }
    }

    // discount one more step to time 0 (from time 1)
    for (int i = 0; i < n_paths; ++i) cashflow[i] *= disc;

    double sum = 0.0;
    for (int i = 0; i < n_paths; ++i) sum += cashflow[i];
    return sum / static_cast<double>(n_paths);
}
