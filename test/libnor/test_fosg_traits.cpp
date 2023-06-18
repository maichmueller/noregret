#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "dummy_classes.hpp"
#include "nor/env/rps.hpp"
#include "nor/factory.hpp"
#include "nor/fosg_traits.hpp"

TEST(fosg_traits, auto_traits)
{
   EXPECT_TRUE((std::is_same_v< auto_action_type< dummy::Traits >, int >) );
   EXPECT_TRUE((std::is_same_v< auto_info_state_type< dummy::Traits >, std::string >) );

   EXPECT_TRUE((std::is_same_v< auto_world_state_type< dummy::Traits >, std::size_t >) );
   EXPECT_TRUE((std::is_same_v< auto_public_state_type< dummy::Traits >, void >) );

   EXPECT_TRUE((std::is_same_v< auto_public_state_type< dummy::Env >, dummy::Publicstate >) );
}

TEST(fosg_traits, partial_match)
{
   EXPECT_TRUE((nor::fosg_traits_partial_match_v< dummy::Traits, dummy::TraitsSuperClass >) );
}

template < typename Sub, typename Super >
   requires nor::fosg_traits_partial_match_v< Sub, Super >
void trait_fosg_partial_match_check()
{
}

TEST(fosg_traits, partial_match_rps)
{
   auto tabular_policy = nor::factory::make_tabular_policy(
      std::unordered_map<
         nor::games::rps::Infostate,
         nor::HashmapActionPolicy< nor::games::rps::Action > >{}
   );

   trait_fosg_partial_match_check< decltype(tabular_policy), nor::games::rps::Environment >();

   EXPECT_TRUE(
      (nor::fosg_traits_partial_match_v< decltype(tabular_policy), nor::games::rps::Environment >)
   );
}
