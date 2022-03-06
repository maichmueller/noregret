#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "nor/nor.hpp"

struct DummyTraitTest {
   using action_type = int;
   using info_state_type = double;
};
namespace nor {
template <>
struct fosg_traits< DummyTraitTest > {
   using action_type = double;
   using info_state_type = size_t;
   using world_state_type = std::string;
};
}  // namespace nor

TEST(fosg_traits, auto_traits)
{
   EXPECT_TRUE(
      (std::is_same_v< typename nor::fosg_auto_traits< DummyTraitTest >::action_type, int >) );
   EXPECT_TRUE(
      (std::
          is_same_v< typename nor::fosg_auto_traits< DummyTraitTest >::info_state_type, double >) );

   EXPECT_TRUE((std::is_same_v<
                typename nor::fosg_auto_traits< DummyTraitTest >::world_state_type,
                std::string >) );
   EXPECT_TRUE(
      (std::
          is_same_v< typename nor::fosg_auto_traits< DummyTraitTest >::public_state_type, void >) );
}

struct DummyClass {
   using action_type = int;
   using info_state_type = double;
};
struct DummySuperClass {
   using action_type = int;
   using info_state_type = double;

   using world_state_type = double;
   using public_state_type = double;
};

TEST(fosg_traits, partial_match)
{
   EXPECT_TRUE((nor::fosg_traits_partial_match_v< DummyClass, DummySuperClass >) );
}
