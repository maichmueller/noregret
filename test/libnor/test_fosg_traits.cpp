#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "nor/nor.hpp"
#include "dummy_classes.hpp"


TEST(fosg_traits, auto_traits)
{
   EXPECT_TRUE(
      (std::is_same_v< typename nor::fosg_auto_traits< dummy::Traits >::action_type, int >) );
   EXPECT_TRUE(
      (std::
          is_same_v< typename nor::fosg_auto_traits< dummy::Traits >::info_state_type, double >) );

   EXPECT_TRUE((std::is_same_v<
                typename nor::fosg_auto_traits< dummy::Traits >::world_state_type,
                std::string >) );
   EXPECT_TRUE(
      (std::
          is_same_v< typename nor::fosg_auto_traits< dummy::Traits >::public_state_type, void >) );

   EXPECT_TRUE(
      (std::
          is_same_v< typename nor::fosg_auto_traits< dummy::Env >::public_state_type, dummy::Env::State >) );
}



TEST(fosg_traits, partial_match)
{
   EXPECT_TRUE((nor::fosg_traits_partial_match_v< dummy::Traits, dummy::TraitsSuperClass >) );
}
