
#ifndef NOR_GAME_DEFS_HPP
#define NOR_GAME_DEFS_HPP

#include <ankerl/unordered_dense.h>

#include <common/string_utils.hpp>

namespace nor {

enum class Player {
   unknown = -2,
   chance = -1,
   alex = 0,
   bob = 1,
   cedric = 2,
   dexter = 3,
   emily = 4,
   florence = 5,
   gustavo = 6,
   henrick = 7,
   ian = 8,
   julia = 9,
   kelvin = 10,
   lea = 11,
   michael = 12,
   norbert = 13,
   oscar = 14,
   pedro = 15,
   quentin = 16,
   rosie = 17,
   sophia = 18,
   tristan = 19,
   ulysses = 20,
   victoria = 21,
   william = 22,
   xavier = 23,
   yusuf = 24,
   zoey = 25
};

template < typename Value >
using player_hashmap = std::unordered_map< Player, Value >;

// template <
//    typename T,
//    class Hash = ankerl::unordered_dense::hash< Player >,
//    class KeyEqual = std::equal_to< Player >,
//    class AllocatorOrContainer = std::allocator< std::pair< Player, T > > >
// using player_hashmap = ankerl::unordered_dense::map< Player, T >;

template <
   class Key,
   class T,
   class Hash = ankerl::unordered_dense::hash< Key >,
   class KeyEqual = std::equal_to< Key >,
   class AllocatorOrContainer = std::allocator< std::pair< Key, T > > >
using dense_hashmap = ankerl::unordered_dense::map< Key, T, Hash, KeyEqual, AllocatorOrContainer >;

enum class Stochasticity {
   deterministic = 0,  // the environment is deterministic
   sample,  // the environment is sampling a random outcome at a given state
   choice  // the environment can provide a vector of random outcomes to choose from at any given
           // state
};

}  // namespace nor

#endif  // NOR_GAME_DEFS_HPP
