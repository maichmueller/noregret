

#ifndef NOR_IS_HPP
#define NOR_IS_HPP

#include <concepts>
#include <cstddef>

namespace nor::concepts::is {

template < typename T >
concept sized = requires(T const t)
{
   {
      t.size()
      } -> std::same_as< size_t >;
};
}  // namespace nor::concepts::is

#endif  // NOR_IS_HPP
