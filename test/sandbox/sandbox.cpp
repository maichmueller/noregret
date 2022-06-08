

#include "sandbox.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <range/v3/all.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "nor/nor.hpp"

template < typename RAContainer, typename Policy >
   requires ranges::range< RAContainer > and requires(Policy p) {
                                                {
                                                   // policy has to be a callable returning the
                                                   // probability of the input matching the
                                                   // container's contained type
                                                   p(std::declval< decltype(*(
                                                        std::declval< RAContainer >().begin())) >())
                                                   } -> std::convertible_to< double >;
                                             }
inline auto& choose(const RAContainer& cont, const Policy& policy, std::mt19937_64& rng)
{
   if constexpr(
      std::random_access_iterator<
         decltype(std::declval< RAContainer >().begin()) > and requires { cont.size(); }) {
      std::vector< double > weights;
      weights.reserve(cont.size());
      for(const auto& elem : cont) {
         weights.emplace_back(policy(elem));
      }
      // the ranges::to_vector method here fails with a segmentation fault for no apparent reason
      //      auto weights = ranges::to_vector(cont | ranges::views::transform(policy));
      return cont[std::discrete_distribution< size_t >(weights.begin(), weights.end())(rng)];
   } else {
      std::vector< double > weights;
      if constexpr(requires { cont.size(); }) {
         // if we can know how many elements are in the container, then reserve that amount.
         weights.reserve(cont.size());
      }
      auto cont_as_vec = ranges::to_vector(cont | ranges::views::transform([&](const auto& elem) {
                                              weights.emplace_back(policy(elem));
                                              return std::ref(elem);
                                           }));
      return cont_as_vec[std::discrete_distribution< size_t >(weights.begin(), weights.end())(rng)]
         .get();
   }
}

int main()
{
   std::vector< int > choices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
   std::vector< double > weights{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
   double sum = 0.;
   for(auto& weight : weights) {
      sum += weight;
   }
   for(auto& weight : weights) {
      weight /= sum;
   }

   auto policy = [&](int i) { return weights[i]; };
   for(auto [value, weight] : ranges::views::zip(choices, weights)) {
      std::cout << "Choice: " << value << ", weight: " << weight << "\n";
   }

   std::unordered_map< int, size_t > counter;
   auto engine = std::mt19937_64{std::random_device{}()};
   size_t n = 10000000;
   for(auto i : ranges::views::iota(size_t(0), n)) {
      counter[choose(choices, policy, engine)] += 1;
   }
   for(auto [value, count] : counter) {
      std::cout << "Value: " << value << ", freq: " << static_cast<double>(count) / n << "\n";
   }
}