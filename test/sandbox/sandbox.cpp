

#include "sandbox.hpp"

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

#include "nor/nor.hpp"
#include "stratego/Logic.hpp"

int main(int argc, char** argv)
{
   std::cout << typeid(std::conditional_t<
                          nor::concepts::has::trait::action_type< stratego::State >,
                          int,
                          double >)
                   .name();
   //   std::map<int,T>{}.at();
}

class std::_Hashtable<
   nor::games::stratego::InfoState,
   std::pair<
      const nor::games::stratego::InfoState,
      std::shared_ptr< nor::rm::CFRNode<
         stratego::Action,
         nor::rm::VanillaCFR<
            nor::rm::CFRConfig{false, false},
            nor::games::stratego::Environment,
            nor::rm::fosg_traits< nor::games::stratego::Environment >,
            nor::TabularPolicy<
               nor::UniformPolicy<
                  nor::games::stratego::InfoState,
                  stratego::Action,
                  nor::HashMapActionPolicy< stratego::Action >,
                  18446744073709551615,
                  MinimalState_vanilla_cfr_usage_stratego_Test::TestBody()::< lambda(
                     const nor::games::stratego::InfoState&) > >,
               nor::games::stratego::InfoState,
               stratego::Action,
               nor::HashMapActionPolicy< stratego::Action >,
               std::unordered_map<
                  nor::games::stratego::InfoState,
                  nor::HashMapActionPolicy< stratego::Action >,
                  std::hash< nor::games::stratego::InfoState >,
                  std::equal_to< nor::games::stratego::InfoState >,
                  std::allocator< std::pair<
                     const nor::games::stratego::InfoState,
                     nor::HashMapActionPolicy< stratego::Action > > > > > >::< lambda() >,
         nor::games::stratego::InfoState > > >,
   std::allocator< std::pair<
      const nor::games::stratego::InfoState,
      std::shared_ptr< nor::rm::CFRNode<
         stratego::Action,
         nor::rm::VanillaCFR<
            nor::rm::CFRConfig{false, false},
            nor::games::stratego::Environment,
            nor::rm::fosg_traits< nor::games::stratego::Environment >,
            nor::TabularPolicy<
               nor::UniformPolicy<
                  nor::games::stratego::InfoState,
                  stratego::Action,
                  nor::HashMapActionPolicy< stratego::Action >,
                  18446744073709551615,
                  MinimalState_vanilla_cfr_usage_stratego_Test::TestBody()::< lambda(
                     const nor::games::stratego::InfoState&) > >,
               nor::games::stratego::InfoState,
               stratego::Action,
               nor::HashMapActionPolicy< stratego::Action >,
               std::unordered_map<
                  nor::games::stratego::InfoState,
                  nor::HashMapActionPolicy< stratego::Action >,
                  std::hash< nor::games::stratego::InfoState >,
                  std::equal_to< nor::games::stratego::InfoState >,
                  std::allocator< std::pair<
                     const nor::games::stratego::InfoState,
                     nor::HashMapActionPolicy< stratego::Action > > > > > >::< lambda() >,
         nor::games::stratego::InfoState > > > >,
   std::__detail::_Select1st,
   std::equal_to< nor::games::stratego::InfoState >,
   std::hash< nor::games::stratego::InfoState >,
   std::__detail::_Mod_range_hashing,
   std::__detail::_Default_ranged_hash,
   std::__detail::_Prime_rehash_policy,
   std::__detail::_Hashtable_traits< false, false, true > >