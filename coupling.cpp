#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace {

struct Atom {
    int index;
    long double mass;
};

struct LowerPriority {
    bool operator()(const Atom& a, const Atom& b) const {
        return a.mass == b.mass ? a.index > b.index : a.mass < b.mass;
    }
};

struct Edge {
    int state;
    int column;
    long double mass;
};

using Heap = std::priority_queue<Atom, std::vector<Atom>, LowerPriority>;

bool higher(const Atom& a, const Atom& b) {
    return LowerPriority{}(b, a);
}

} // namespace

// Returns the number of nonzero entries in the chosen conditional posterior.
// A negative observed column selects encoding mode. Output storage holds at
// most one entry per input state. All probabilities are normalized internally.
extern "C" int d203_couple(
    int count, const int32_t* states, const double* prior,
    int width, const double* channel, int true_state, double coin, int observed,
    int* token, int32_t* out_states, double* out_probs, double* out_entropy
) {
    std::vector<Atom> rows;
    rows.reserve(count);
    long double prior_sum = 0, channel_sum = 0, true_mass = 0;
    for (int i = 0; i < count; ++i) {
        const long double mass = prior[i];
        prior_sum += mass;
        if (states[i] == true_state) true_mass = mass;
        rows.push_back({states[i], mass});
    }
    if (observed < 0 && true_mass == 0) return -1;
    if (!std::is_sorted(rows.begin(), rows.end(), higher)) {
        std::sort(rows.begin(), rows.end(), higher);
    }
    for (int j = 0; j < width; ++j) channel_sum += channel[j];
    Heap columns, updated_rows;
    for (int j = 0; j < width; ++j) {
        if (channel[j] > 0) {
            columns.push({j, channel[j] * (prior_sum / channel_sum)});
        }
    }

    // Unmodified rows are consumed in sorted order. Only rows that have
    // received a partial allocation need to enter the heap.
    std::size_t next_row = 0;
    std::vector<Edge> edges;
    if (observed < 0) edges.reserve(count + width);
    int chosen = observed, last_true_column = -1, used = 0;
    long double remaining_coin = coin * true_mass, posterior_sum = 0;
    while (!columns.empty() && (next_row < rows.size() || !updated_rows.empty())) {
        Atom row;
        if (!updated_rows.empty() &&
            (next_row == rows.size() || higher(updated_rows.top(), rows[next_row]))) {
            row = updated_rows.top();
            updated_rows.pop();
        } else {
            row = rows[next_row++];
        }
        Atom col = columns.top();
        columns.pop();
        const long double mass = std::min(row.mass, col.mass);
        row.mass -= mass;
        col.mass -= mass;
        if (row.mass > 0) updated_rows.push(row);
        if (col.mass > 0) columns.push(col);

        if (observed < 0) {
            edges.push_back({row.index, col.index, mass});
            if (row.index == true_state) {
                last_true_column = col.index;
                if (chosen < 0 && remaining_coin < mass) chosen = col.index;
                remaining_coin -= mass;
            }
        } else if (col.index == observed) {
            out_states[used] = row.index;
            out_probs[used++] = mass;
            posterior_sum += mass;
        }
    }

    if (observed < 0) {
        if (chosen < 0) chosen = last_true_column;
        if (chosen < 0) return -2;
        for (const auto& edge : edges) {
            if (edge.column != chosen) continue;
            out_states[used] = edge.state;
            out_probs[used++] = edge.mass;
            posterior_sum += edge.mass;
        }
    }
    if (used == 0) return -3;
    if (std::fabs(posterior_sum / prior_sum - channel[chosen] / channel_sum) > 1e-12L) {
        return -4;
    }

    // A greedy coupling visits each (row, column) pair at most once because
    // every allocation exhausts one of the two current residuals.
    double entropy = 0;
    double best_mass = -1;
    int best_state = -1;
    for (int i = 0; i < used; ++i) {
        const double p = out_probs[i] / posterior_sum;
        out_probs[i] = p;
        entropy -= p * std::log2(p);
        if (p > best_mass || (p == best_mass && out_states[i] < best_state)) {
            best_mass = p;
            best_state = out_states[i];
        }
    }
    token[0] = chosen;
    token[1] = best_state;
    *out_entropy = entropy;
    return used;
}
