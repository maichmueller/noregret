#include <gtest/gtest.h>

#include "dummy_classes.hpp"
#include "nor/concepts.hpp"
#include "nor/env.hpp"
#include "nor/policy/policy.hpp"

TEST(concrete, iterable)
{
   EXPECT_TRUE((nor::concepts::iterable< std::vector< int > >) );
   EXPECT_TRUE((nor::concepts::iterable< std::vector< double > >) );
   EXPECT_TRUE((nor::concepts::iterable< std::map< int, int > >) );
   EXPECT_TRUE((nor::concepts::iterable< std::unordered_map< int, int > >) );
   EXPECT_TRUE((nor::concepts::iterable< std::string >) );
   // custom types
   EXPECT_TRUE((nor::concepts::iterable< nor::HashmapActionPolicy< int > >) );
}

TEST(concrete, sized)
{
   EXPECT_TRUE((nor::concepts::is::sized< std::vector< int > >) );
   EXPECT_TRUE((nor::concepts::is::sized< std::vector< double > >) );
   EXPECT_TRUE((nor::concepts::is::sized< std::map< int, int > >) );
   EXPECT_TRUE((nor::concepts::is::sized< std::unordered_map< int, int > >) );
   EXPECT_TRUE((nor::concepts::is::sized< std::string >) );
   // custom types
   EXPECT_TRUE((nor::concepts::is::sized< nor::HashmapActionPolicy< int > >) );
}

TEST(concrete, action_policy)
{
   EXPECT_TRUE((nor::concepts::action_policy< nor::HashmapActionPolicy< int > >) );
   EXPECT_TRUE(
      (nor::concepts::action_policy< nor::HashmapActionPolicy< nor::games::stratego::Action > >) );
}

template < typename Policy, typename Infostate, typename Action >
   requires nor::concepts::reference_state_policy< Policy, Infostate, Action >
void concept_state_policy_check();

TEST(concrete, memorizing_state_policy)
{
   //   concept_state_policy_check<
   //      nor::TabularPolicy< dummy::Infostate, nor::HashmapActionPolicy< int > >,
   //      dummy::Infostate,
   //      typename nor::fosg_auto_traits< dummy::Infostate >::observation_type >();

   EXPECT_TRUE((nor::concepts::reference_state_policy<
                nor::TabularPolicy<
                   dummy::Infostate,
                   nor::HashmapActionPolicy< int >,
                   nor::UniformPolicy< dummy::Infostate, nor::HashmapActionPolicy< int > > >,
                dummy::Infostate,
                int >) );
}

template < typename Policy, typename Infostate, typename ActionPolicy >
   requires nor::concepts::default_state_policy< Policy, Infostate, ActionPolicy >
void concept_default_state_policy_check();

TEST(concrete, default_state_policy)
{
   //   concept_default_state_policy_check<
   //      nor::UniformPolicy< dummy::Infostate, nor::HashmapActionPolicy< int > >,
   //      dummy::Infostate,
   //      nor::HashmapActionPolicy< int > >();

   EXPECT_TRUE((nor::concepts::default_state_policy<
                nor::UniformPolicy< dummy::Infostate, nor::HashmapActionPolicy< int > >,
                dummy::Infostate,
                nor::HashmapActionPolicy< int > >) );
}

template < typename Env >
   requires nor::concepts::fosg< Env >
void concept_fosg_check();

template < typename Env >
   requires nor::concepts::deterministic_fosg< Env >
void concept_deterministic_fosg_check();

TEST(concrete, fosg_dummy)
{
//         concept_fosg_check< dummy::Env >();

   EXPECT_TRUE((nor::concepts::fosg< dummy::Env >) );
}


TEST(concrete, fosg_kuhn)
{
   EXPECT_TRUE((nor::concepts::fosg< nor::games::kuhn::Environment >) );
   EXPECT_FALSE((nor::concepts::deterministic_fosg< nor::games::kuhn::Environment >) );
}

TEST(concrete, fosg_stratego)
{
   //   concept_fosg_check< nor::games::stratego::Environment >();
   //   concept_deterministic_fosg_check< nor::games::stratego::Environment >();

   EXPECT_TRUE((nor::concepts::fosg< nor::games::stratego::Environment >) );
   EXPECT_TRUE((nor::concepts::deterministic_fosg< nor::games::stratego::Environment >) );
}

TEST(concrete, vanilla_requirements) {}
