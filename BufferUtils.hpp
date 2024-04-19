#pragma once

#include <random>
#include <algorithm>
#include <type_traits>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::normal_distribution<float> dist(-10.f, 10.f);

template<typename Iter>
void randomize(Iter beg, Iter end) {
    using ValueType = typename std::remove_pointer<Iter>::type;
    for (auto i = beg; i != end; ++i) {
        auto tmp = dist(gen);
        tmp = std::max(-100.0f, tmp);
        tmp = std::min(100.0f, tmp);
        *i = tmp;
    }
}
