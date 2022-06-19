#ifndef NOR_PLAYER_VECTOR_HPP
#define NOR_PLAYER_VECTOR_HPP

#include <vector>

template <typename T, typename Allocator = std::allocator< T>>
class PlayerVector : std::vector< T, Allocator > {
   using base = std::vector< T, Allocator >;

  public:

   // inherit constructors
   using base::base;

   // inherit existing API

};

#endif  // NOR_PLAYER_VECTOR_HPP
