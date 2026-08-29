#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "battleship/environment.hpp"
#include "common/common.hpp"
#include "goofspiel/environment.hpp"
#include "liars_dice/environment.hpp"
#include "nor/game_defs.hpp"
#include "nor/env.hpp"
#include "nor/fosg_traits.hpp"
#include "nor/meta/auto_types.hpp"
#include "nor/meta/diagnostics.hpp"
#include "nor/meta/enum_names.hpp"
#include "nor/meta/features.hpp"
#include "nor/utils/utils.hpp"
#include "texas_holdem_poker/environment.hpp"

// The expected names pin down the spelling of the reflected enum declarations.

namespace {

constexpr std::array< nor::Player, 28 > all_players{
   nor::Player::unknown, nor::Player::chance,  nor::Player::alex,    nor::Player::bob,
   nor::Player::cedric,  nor::Player::dexter,  nor::Player::emily,   nor::Player::florence,
   nor::Player::gustavo, nor::Player::henrick, nor::Player::ian,     nor::Player::julia,
   nor::Player::kelvin,  nor::Player::lea,     nor::Player::michael, nor::Player::norbert,
   nor::Player::oscar,   nor::Player::pedro,   nor::Player::quentin, nor::Player::rosie,
   nor::Player::sophia,  nor::Player::tristan, nor::Player::ulysses, nor::Player::victoria,
   nor::Player::william, nor::Player::xavier,  nor::Player::yusuf,   nor::Player::zoey};

constexpr std::array< std::string_view, 28 > player_golden_names{
   "unknown",  "chance",  "alex",     "bob",     "cedric",  "dexter", "emily",
   "florence", "gustavo", "henrick",  "ian",     "julia",   "kelvin", "lea",
   "michael",  "norbert", "oscar",    "pedro",   "quentin", "rosie",  "sophia",
   "tristan",  "ulysses", "victoria", "william", "xavier",  "yusuf",  "zoey"};

constexpr std::array< nor::Stochasticity, 3 > all_stochasticities{
   nor::Stochasticity::deterministic,
   nor::Stochasticity::sample,
   nor::Stochasticity::choice};

constexpr std::array< std::string_view, 3 > stochasticity_golden_names{
   "deterministic",
   "sample",
   "choice"};

enum class ReflectionTestEnum { first = -1, second = 7 };

struct ReflectedBase {
   using observation_type = long;
};

struct ReflectedDerived: ReflectedBase {};

}  // namespace

static_assert(
   nor::meta::enum_name(nor::Player::alex) == "alex",
   "enum_name must be usable in constant expressions"
);
static_assert(
   nor::meta::enum_from_name< nor::Player >("chance") == nor::Player::chance,
   "enum_from_name must be usable in constant expressions"
);
static_assert(nor::meta::enum_name(ReflectionTestEnum::first) == "first");
static_assert(
   nor::meta::enum_from_name< ReflectionTestEnum >("second") == ReflectionTestEnum::second
);
static_assert(
   std::same_as< nor::meta::auto_observation_type< ReflectedDerived >, long >,
   "associated type lookup must include inherited reflected members"
);
static_assert(
   std::same_as< nor::auto_observation_type< ReflectedDerived >, long >,
   "the established auto_* aliases must use reflected lookup"
);
static_assert(
   std::same_as< typename nor::fosg_traits< ReflectedDerived >::observation_type, long >,
   "the compatibility fosg_traits view must use reflected lookup"
);

TEST(MetaEnumNames, player_names_match_golden_strings)
{
   for(auto i = 0u; i < all_players.size(); ++i) {
      EXPECT_EQ(nor::meta::enum_name(all_players[i]), player_golden_names[i]);
   }
}

TEST(MetaEnumNames, stochasticity_names_match_golden_strings)
{
   for(auto i = 0u; i < all_stochasticities.size(); ++i) {
      EXPECT_EQ(nor::meta::enum_name(all_stochasticities[i]), stochasticity_golden_names[i]);
   }
}

TEST(MetaEnumNames, player_roundtrip_over_all_enumerators)
{
   for(auto i = 0u; i < all_players.size(); ++i) {
      const auto back = nor::meta::enum_from_name< nor::Player >(player_golden_names[i]);
      ASSERT_TRUE(back.has_value());
      EXPECT_EQ(*back, all_players[i]);
      // and the inverse direction
      const auto name = nor::meta::enum_from_name< nor::Player >(nor::meta::enum_name(all_players[i]
      ));
      ASSERT_TRUE(name.has_value());
      EXPECT_EQ(*name, all_players[i]);
   }
}

TEST(MetaEnumNames, stochasticity_roundtrip_over_all_enumerators)
{
   for(auto i = 0u; i < all_stochasticities.size(); ++i) {
      const auto back = nor::meta::enum_from_name< nor::Stochasticity >(
         stochasticity_golden_names[i]
      );
      ASSERT_TRUE(back.has_value());
      EXPECT_EQ(*back, all_stochasticities[i]);
   }
}

TEST(MetaEnumNames, unknown_values_have_no_name)
{
   EXPECT_TRUE(nor::meta::enum_name(static_cast< nor::Player >(99)).empty());
   EXPECT_TRUE(nor::meta::enum_name(static_cast< nor::Stochasticity >(42)).empty());
   EXPECT_FALSE(nor::meta::enum_from_name< nor::Player >("this_player_does_not_exist").has_value());
   EXPECT_FALSE(nor::meta::enum_from_name< nor::Stochasticity >("chaotic").has_value());
}

TEST(MetaStringUtils, to_string_parity_with_golden_tables)
{
   EXPECT_EQ(common::to_string(nor::Player::chance), "chance");
   EXPECT_EQ(common::to_string(nor::Player::alex), "alex");
   EXPECT_EQ(common::to_string(nor::Player::bob), "bob");
   EXPECT_EQ(common::to_string(nor::Player::michael), "michael");
   EXPECT_EQ(common::to_string(nor::Player::zoey), "zoey");
   EXPECT_EQ(common::to_string(nor::Player::unknown), "unknown");

   EXPECT_EQ(common::to_string(nor::Stochasticity::deterministic), "deterministic");
   EXPECT_EQ(common::to_string(nor::Stochasticity::sample), "sample");
   EXPECT_EQ(common::to_string(nor::Stochasticity::choice), "choice");
}

TEST(MetaStringUtils, from_string_parity_with_golden_tables)
{
   EXPECT_EQ(common::from_string< nor::Player >("chance"), nor::Player::chance);
   EXPECT_EQ(common::from_string< nor::Player >("alex"), nor::Player::alex);
   EXPECT_EQ(common::from_string< nor::Player >("zoey"), nor::Player::zoey);
   EXPECT_EQ(common::from_string< nor::Player >("unknown"), nor::Player::unknown);
   EXPECT_THROW(common::from_string< nor::Player >("nonsense"), std::range_error);
}

// The enum concat operator+ helpers require heterogeneous string/string_view
// concatenation (P2591), provided by GCC 16's C++26 library.
TEST(MetaStringUtils, concat_helpers)
{
   EXPECT_EQ(std::string_view("p_") + nor::Player::bob, "p_bob");
   EXPECT_EQ(std::string("prefix_") + nor::Player::chance, "prefix_chance");
   EXPECT_EQ(nor::Player::bob + std::string_view("!"), "bob!");
   EXPECT_EQ(nor::Stochasticity::sample + "_suffix", "sample_suffix");
}

namespace {

/**
 * @brief Compile-time proof that the public auto_* aliases resolve through
 * reflected associated types on every registered environment type.
 */
template < typename Env >
concept canonical_reflected_aliases =
   std::same_as< nor::auto_action_type< Env >, nor::meta::fosg_type_or_void< Env, "action_type" > >
   && std::
      same_as< nor::auto_chance_outcome_type< Env >,
               nor::meta::fosg_type_or_void< Env, "chance_outcome_type" > >
   && std::same_as<
      nor::auto_action_policy_type< Env >,
      nor::meta::fosg_type_or_void< Env, "action_policy_type" > >
   && std::same_as<
      nor::auto_chance_distribution_type< Env >,
      nor::meta::fosg_type_or_void< Env, "chance_distribution_type" > >
   && std::same_as<
      nor::auto_observation_type< Env >,
      nor::meta::fosg_type_or_void< Env, "observation_type" > >
   && std::same_as<
      nor::auto_info_state_type< Env >,
      nor::meta::fosg_type_or_void< Env, "info_state_type" > >
   && std::same_as<
      nor::auto_public_state_type< Env >,
      nor::meta::fosg_type_or_void< Env, "public_state_type" > >
   && std::same_as<
      nor::auto_world_state_type< Env >,
      nor::meta::fosg_type_or_void< Env, "world_state_type" > >;

template < typename... Envs >
consteval bool all_agree()
{
   return (canonical_reflected_aliases< Envs > and ...);
}

static_assert(all_agree<
              nor::games::kuhn::Environment,
              nor::games::leduc::Environment,
              nor::games::rps::Environment,
              nor::games::stratego::Environment,
              nor::games::texholdem::Environment,
              nor::games::goofspiel::Environment,
              nor::games::liars_dice::Environment,
              nor::games::battleship::Environment >());

static_assert(
   std::same_as<
      nor::auto_observation_type< nor::games::kuhn::Infostate >,
      nor::games::kuhn::Observation >
);
static_assert(
   std::same_as<
      nor::auto_observation_type< nor::games::kuhn::Publicstate >,
      nor::games::kuhn::Observation >
);

struct IncompleteEnv {};

static_assert(not nor::meta::has_all_fosg_members< IncompleteEnv >());
static_assert(nor::meta::missing_fosg_members< IncompleteEnv >().count == 6);
static_assert(
   nor::meta::missing_fosg_members_message< IncompleteEnv >().view()
   == "missing FOSG member type(s): action_type, observation_type, info_state_type, "
      "public_state_type, world_state_type, chance_outcome_type"
);
static_assert(
   nor::meta::missing_fosg_members_message< nor::games::kuhn::Environment >().view()
   == "all FOSG member types present"
);

}  // namespace

TEST(MetaDiagnostics, complete_environments_report_no_missing_members)
{
   static_assert(nor::meta::has_all_fosg_members< nor::games::kuhn::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::leduc::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::rps::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::stratego::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::texholdem::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::goofspiel::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::liars_dice::Environment >());
   static_assert(nor::meta::has_all_fosg_members< nor::games::battleship::Environment >());
   SUCCEED();
}

TEST(MetaDiagnostics, incomplete_struct_yields_expected_message)
{
   EXPECT_EQ(
      nor::meta::missing_fosg_members_message< IncompleteEnv >().view(),
      "missing FOSG member type(s): action_type, observation_type, info_state_type, "
      "public_state_type, world_state_type, chance_outcome_type"
   );
}
