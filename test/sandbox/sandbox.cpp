

#include "sandbox.hpp"

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

#include "nor/nor.hpp"

namespace detail {
template < uint8_t... digits >
struct positive_to_chars {
   static const char value[];
};
template < uint8_t... digits >
const char positive_to_chars< digits... >::value[] = {('0' + digits)..., 0};

template < uint8_t... digits >
struct negative_to_chars {
   static const char value[];
};
template < uint8_t... digits >
const char negative_to_chars< digits... >::value[] = {'-', ('0' + digits)..., 0};

template < bool neg, uint8_t... digits >
struct to_chars: positive_to_chars< digits... > {
};

template < uint8_t... digits >
struct to_chars< true, digits... >: negative_to_chars< digits... > {
};

template < bool neg, uintmax_t rem, uint8_t... digits >
struct explode: explode< neg, rem / 10, rem % 10, digits... > {
};

template < bool neg, uint8_t... digits >
struct explode< neg, 0, digits... >: to_chars< neg, digits... > {
};

template < typename T >
consteval uintmax_t cabs(T num)
{
   return (num < 0) ? -num : num;
}
}  // namespace detail

template < typename Integer, Integer num >
struct string_from: detail::explode< (num < 0), detail::cabs(num) > {
};

int main(int argc, char** argv)
{
   std::cout << string_from< int, 45 >::value << '\n' << string_from< int, -1 >::value << '\n';
}
