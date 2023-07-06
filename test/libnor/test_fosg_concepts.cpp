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
   using namespace nor;
   using namespace games::stratego;

   EXPECT_TRUE((nor::concepts::action_policy< HashmapActionPolicy< int > >) );
   EXPECT_TRUE((nor::concepts::action_policy< HashmapActionPolicy< Action > >) );
}

template <
   typename T,
   typename Infostate,
   typename Action,
   typename ActionPolicy = nor::auto_action_policy_type< T > >
   requires nor::concepts::default_state_policy< T, Infostate, Action, ActionPolicy >
void concept_default_state_policy_check()
{
}

TEST(concrete, default_state_policy)
{
   using namespace nor;
   using namespace dummy;

   concept_default_state_policy_check<
      UniformPolicy< Infostate, HashmapActionPolicy< int > >,
      Infostate,
      int,
      HashmapActionPolicy< int > >();

   EXPECT_TRUE((nor::concepts::default_state_policy<
                UniformPolicy< Infostate, HashmapActionPolicy< int > >,
                Infostate,
                int >) );
}

template <
   typename Policy,
   typename DefaultPolicy,
   typename Infostate,
   typename Action,
   typename ActionPolicy >
   requires nor::concepts::
      reference_state_policy< Policy, DefaultPolicy, Infostate, Action, ActionPolicy >
   void concept_reference_state_policy_check()
{
}

TEST(concrete, referencing_state_policy)
{
   using namespace nor;
   using namespace dummy;

   static_assert(std::is_same_v<
                 typename TabularPolicy< Infostate, HashmapActionPolicy< int > >::action_type,
                 int >);
   static_assert(std::is_same_v<
                 typename TabularPolicy< Infostate, HashmapActionPolicy< int > >::
                    action_policy_type,
                 HashmapActionPolicy< int > >);

   concept_reference_state_policy_check<
      TabularPolicy< Infostate, HashmapActionPolicy< int > >,
      UniformPolicy< Infostate, HashmapActionPolicy< int > >,
      Infostate,
      int,
      HashmapActionPolicy< int > >();

   EXPECT_TRUE((nor::concepts::reference_state_policy<
                TabularPolicy< Infostate, HashmapActionPolicy< int > >,
                UniformPolicy< Infostate, HashmapActionPolicy< int > >,
                Infostate,
                int >) );
}

template < typename Env >
   requires nor::concepts::fosg< Env >
void concept_fosg_check()
{
}

template < typename Env >
   requires nor::concepts::deterministic_fosg< Env >
void concept_deterministic_fosg_check()
{
}

TEST(concrete, fosg_dummy)
{
   using namespace dummy;
   concept_fosg_check< Env<> >();

   EXPECT_TRUE((nor::concepts::fosg< Env<> >) );
}

TEST(concrete, fosg_kuhn)
{
   using namespace nor::games::kuhn;
   EXPECT_TRUE((nor::concepts::fosg< Environment >) );
   EXPECT_FALSE((nor::concepts::deterministic_fosg< Environment >) );
}

TEST(concrete, fosg_leduc)
{
   //   EXPECT_TRUE((nor::concepts::fosg< nor::games::leduc::Environment >) );
   //   EXPECT_FALSE((nor::concepts::deterministic_fosg< nor::games::leduc::Environment >) );
}

TEST(concrete, fosg_polymorphic)
{
   using namespace nor::games::polymorph;
   concept_fosg_check< Environment >();
   //   concept_deterministic_fosg_check< nor::games::polymorph::Environment >();

   EXPECT_TRUE((nor::concepts::fosg< Environment >) );
   EXPECT_FALSE((nor::concepts::deterministic_fosg< Environment >) );
}

TEST(concrete, fosg_stratego)
{
   using namespace nor::games::stratego;
   concept_fosg_check< Environment >();
   concept_deterministic_fosg_check< Environment >();

   EXPECT_TRUE((nor::concepts::fosg< Environment >) );
   EXPECT_TRUE((nor::concepts::deterministic_fosg< Environment >) );
}

TEST(concrete, vanilla_requirements) {}
