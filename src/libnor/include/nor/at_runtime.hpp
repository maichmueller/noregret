
#ifndef NOR_AT_RUNTIME_HPP
#define NOR_AT_RUNTIME_HPP

#include <exception>

namespace nor {

template < typename Env >
constexpr void assert_serialized_and_unrolled(const Env& e)
{
   if(not e.serialized()) {
      throw std::invalid_argument("The environment needs to be a serialized and unrolled fosg.");
   }
}

}  // namespace nor

#endif  // NOR_AT_RUNTIME_HPP
