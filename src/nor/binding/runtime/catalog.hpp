#ifndef NOR_BINDING_RUNTIME_CATALOG_HPP
#define NOR_BINDING_RUNTIME_CATALOG_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "battleship/environment.hpp"
#include "battleship_gs/environment.hpp"
#include "centipede/environment.hpp"
#include "colonel_blotto/environment.hpp"
#include "dark_hex/environment.hpp"
#include "goofspiel/environment.hpp"
#include "liars_dice/environment.hpp"
#include "nor/env/kuhn.hpp"
#include "nor/env/leduc.hpp"
#include "nor/env/rps.hpp"
#include "nor/env/stratego.hpp"
#include "nor/rm.hpp"
#include "oshi_zumo/environment.hpp"
#include "pursuit_evasion/environment.hpp"
#include "shapley/environment.hpp"
#include "sheriff/environment.hpp"
#include "texas_holdem_poker/environment.hpp"
#include "three_player_goofspiel/environment.hpp"
#include "types.hpp"

namespace nor::binding::runtime {
namespace detail {

template < typename... Ts >
struct type_list {};

template < typename Value, size_t Count >
[[nodiscard]] consteval bool unique_values(const std::array< Value, Count >& values)
{
   for(size_t left = 0; left < Count; ++left) {
      for(size_t right = left + 1; right < Count; ++right) {
         if(values[left] == values[right])
            return false;
      }
   }
   return true;
}

template < size_t Count >
[[nodiscard]] consteval bool unique_field_ids(const std::array< FieldDescriptor, Count >& fields)
{
   std::array< GameFieldId, Count > ids{};
   for(size_t index = 0; index < Count; ++index)
      ids[index] = fields[index].id;
   return unique_values(ids);
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// game specifications //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Tag >
[[nodiscard]] CapabilityError invalid_spec_error(std::string message, GameId game = Tag::id)
{
   return CapabilityError{
      .code = CapabilityErrorCode::invalid_spec, .message = std::move(message), .game = game};
}

[[nodiscard]] inline bool spec_kind_matches(SpecKind kind, const SpecValue& value)
{
   switch(kind) {
      case SpecKind::unsigned_integer: return std::holds_alternative< uint64_t >(value);
      case SpecKind::floating_point:
         return std::holds_alternative< double >(value)
                or std::holds_alternative< uint64_t >(value);
      case SpecKind::boolean: return std::holds_alternative< bool >(value);
   }
   return false;
}

template < typename Tag >
[[nodiscard]] Result< GameSpec > normalized_spec(const GameSpec& input)
{
   if(input.game_id() != Tag::id) {
      return std::unexpected(
         invalid_spec_error< Tag >("GameSpec ID does not match its catalog entry.")
      );
   }

   for(const auto& supplied : input.fields()) {
      const auto found = std::ranges::find(Tag::fields, supplied.id, &FieldDescriptor::id);
      if(found == Tag::fields.end()) {
         return std::unexpected(invalid_spec_error< Tag >(
            "field is not accepted by game '" + std::string(Tag::name) + "'."
         ));
      }
      if(not spec_kind_matches(found->kind, supplied.value)) {
         return std::unexpected(invalid_spec_error< Tag >(
            "field '" + std::string(found->name) + "' has the wrong value kind."
         ));
      }
      if(found->kind == SpecKind::floating_point) {
         if(const auto* floating = std::get_if< double >(&supplied.value);
            floating != nullptr and not std::isfinite(*floating)) {
            return std::unexpected(
               invalid_spec_error< Tag >("field '" + std::string(found->name) + "' must be finite.")
            );
         }
      }
   }

   GameSpec normalized = input;
   for(const auto& field : Tag::fields) {
      if(not normalized.contains(field.id))
         normalized.set(field.id, field.default_value);
   }
   return normalized;
}

[[nodiscard]] inline uint64_t spec_unsigned(const GameSpec& spec, GameFieldId field)
{
   const auto* value = spec.find(field);
   if(value == nullptr)
      throw std::invalid_argument("required unsigned GameSpec field is absent");
   return std::get< uint64_t >(*value);
}

template < std::unsigned_integral Integer >
[[nodiscard]] Integer spec_unsigned_as(const GameSpec& spec, GameFieldId field)
{
   const auto value = spec_unsigned(spec, field);
   if(value > static_cast< uint64_t >(std::numeric_limits< Integer >::max())) {
      throw std::invalid_argument("unsigned GameSpec field exceeds its concrete range");
   }
   return static_cast< Integer >(value);
}

[[nodiscard]] inline double spec_floating(const GameSpec& spec, GameFieldId field)
{
   const auto* value = spec.find(field);
   if(value == nullptr)
      throw std::invalid_argument("required floating GameSpec field is absent");
   if(const auto* floating = std::get_if< double >(value))
      return *floating;
   return static_cast< double >(std::get< uint64_t >(*value));
}

[[nodiscard]] inline bool spec_boolean(const GameSpec& spec, GameFieldId field)
{
   const auto* value = spec.find(field);
   if(value == nullptr)
      throw std::invalid_argument("required boolean GameSpec field is absent");
   return std::get< bool >(*value);
}

template < typename Env >
[[nodiscard]] constexpr size_t default_min_players()
{
   if constexpr(Env::player_count() == std::dynamic_extent)
      return 2;
   return Env::player_count();
}

template < typename Env >
[[nodiscard]] constexpr size_t default_max_players()
{
   if constexpr(Env::player_count() == std::dynamic_extent)
      return Env::max_player_count();
   return Env::player_count();
}

// The standalone wrapper environments used by the existing C++ game targets
// expose their initial state as a default-constructed world state, while the
// richer environment types expose initial_world_state(). Keep that difference
// inside the binding-runtime adapter so neither the registry nor libnor needs
// a special-case API.
template < typename Game >
[[nodiscard]] auto initial_world_state(const typename Game::env_type& environment)
{
   using env_type = typename Game::env_type;
   if constexpr(requires(const env_type& env) { env.initial_world_state(); }) {
      return environment.initial_world_state();
   } else if constexpr(requires { Game::default_world_state(); }) {
      return Game::default_world_state();
   } else {
      return typename env_type::world_state_type{};
   }
}

// Each tag is a single source of truth for a stable ID, its fields, and its concrete environment
// constructor.  The arrays below are intentionally explicit: adding a game requires adding one
// tag here, after which descriptor and capability generation remains generic.
struct kuhn_game {
   using env_type = games::kuhn::Environment;
   static constexpr GameId id = GameId::kuhn;
   static constexpr std::string_view name = "kuhn_poker";
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct leduc_game {
   using env_type = games::leduc::Environment;
   static constexpr GameId id = GameId::leduc;
   static constexpr std::string_view name = "leduc_poker";
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct rps_game {
   using env_type = games::rps::Environment;
   static constexpr GameId id = GameId::rps;
   static constexpr std::string_view name = "rock_paper_scissors";
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct stratego_game {
   using env_type = games::stratego::Environment;
   static constexpr GameId id = GameId::stratego;
   static constexpr std::string_view name = "stratego";
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }

   static env_type::world_state_type default_world_state()
   {
      // The legacy Stratego convenience constructor currently delegates to a
      // broken default-start-field helper.  Supplying empty, explicit setups
      // keeps this binding adapter constructible without changing the game
      // implementation; the later binding can add a richer Stratego spec when
      // that environment exposes one.
      using setup_map = std::map< ::stratego::Team, std::optional< ::stratego::Config::setup_t > >;
      const setup_map empty_setups{
         {::stratego::Team::BLUE, ::stratego::Config::setup_t{}},
         {::stratego::Team::RED, ::stratego::Config::setup_t{}}};
      return env_type::world_state_type{
         ::stratego::Config{::stratego::Team::BLUE, size_t{5}, empty_setups}};
   }
};

struct texas_holdem_game {
   using env_type = games::texholdem::Environment;
   static constexpr GameId id = GameId::texas_holdem;
   static constexpr std::string_view name = "texas_holdem";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::n_players,
         "n_players",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::starting_stack,
         "starting_stack",
         SpecKind::floating_point,
         SpecValue{200.}},
      FieldDescriptor{
         GameFieldId::small_blind,
         "small_blind",
         SpecKind::floating_point,
         SpecValue{1.}},
      FieldDescriptor{
         GameFieldId::big_blind,
         "big_blind",
         SpecKind::floating_point,
         SpecValue{2.}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{::texholdem::PokerConfig{
         spec_unsigned_as< size_t >(spec, GameFieldId::n_players),
         spec_floating(spec, GameFieldId::starting_stack),
         spec_floating(spec, GameFieldId::small_blind),
         spec_floating(spec, GameFieldId::big_blind)}};
   }
};

struct goofspiel_game {
   using env_type = games::goofspiel::Environment;
   static constexpr GameId id = GameId::goofspiel;
   static constexpr std::string_view name = "goofspiel";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::deck_size,
         "deck_size",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{GameFieldId::imp_info, "imp_info", SpecKind::boolean, SpecValue{false}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      auto config = ::goofspiel::GoofspielConfig{
         .deck_size = spec_unsigned_as< size_t >(spec, GameFieldId::deck_size),
         .imp_info = spec_boolean(spec, GameFieldId::imp_info)};
      config.validate();
      return env_type{std::move(config)};
   }
};

struct three_player_goofspiel_game {
   using env_type = games::three_player_goofspiel::Environment;
   static constexpr GameId id = GameId::three_player_goofspiel;
   static constexpr std::string_view name = "three_player_goofspiel";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::deck_size,
         "deck_size",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{GameFieldId::imp_info, "imp_info", SpecKind::boolean, SpecValue{false}},
      FieldDescriptor{
         GameFieldId::split_half_deal,
         "split_half_deal",
         SpecKind::boolean,
         SpecValue{false}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      auto config = ::three_player_goofspiel::GoofspielConfig{
         .deck_size = spec_unsigned_as< size_t >(spec, GameFieldId::deck_size),
         .imp_info = spec_boolean(spec, GameFieldId::imp_info),
         .split_half_deal = spec_boolean(spec, GameFieldId::split_half_deal)};
      config.validate();
      return env_type{std::move(config)};
   }
};

struct battleship_game {
   using env_type = games::battleship::Environment;
   static constexpr GameId id = GameId::battleship;
   static constexpr std::string_view name = "battleship";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::rows,
         "rows",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::cols,
         "cols",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::ships_per_fleet,
         "ships_per_fleet",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{1}}},
      FieldDescriptor{
         GameFieldId::max_shots,
         "max_shots",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{
         GameFieldId::ship_value,
         "ship_value",
         SpecKind::floating_point,
         SpecValue{2.}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{::battleship::Config{
         spec_unsigned_as< size_t >(spec, GameFieldId::rows),
         spec_unsigned_as< size_t >(spec, GameFieldId::cols),
         spec_unsigned_as< size_t >(spec, GameFieldId::ships_per_fleet),
         spec_unsigned_as< size_t >(spec, GameFieldId::max_shots),
         spec_floating(spec, GameFieldId::ship_value)}};
   }
};

struct battleship_gs_game {
   using env_type = games::battleship_gs::Environment;
   static constexpr GameId id = GameId::battleship_gs;
   static constexpr std::string_view name = "battleship_gs";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::rows,
         "rows",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{
         GameFieldId::cols,
         "cols",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{1}}},
      FieldDescriptor{
         GameFieldId::max_shots,
         "max_shots",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::loss_multiplier,
         "loss_multiplier",
         SpecKind::floating_point,
         SpecValue{2.}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      auto config = ::battleship_gs::Config{
         spec_unsigned_as< size_t >(spec, GameFieldId::rows),
         spec_unsigned_as< size_t >(spec, GameFieldId::cols),
         std::vector< ::battleship_gs::ShipSpec >{{uint8_t{1}, 1.}},
         spec_unsigned_as< size_t >(spec, GameFieldId::max_shots),
         spec_floating(spec, GameFieldId::loss_multiplier)};
      return env_type{std::move(config)};
   }
};

struct dark_hex_game {
   using env_type = games::dark_hex::Environment;
   static constexpr GameId id = GameId::dark_hex;
   static constexpr std::string_view name = "dark_hex";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::board_size,
         "board_size",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{
         GameFieldId::rules_mode,
         "rules_mode",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{0}}},
      FieldDescriptor{
         GameFieldId::move_limit,
         "move_limit",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{0}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      const auto rules = spec_unsigned(spec, GameFieldId::rules_mode);
      if(rules > 1)
         throw std::invalid_argument("dark_hex rules_mode must be 0 (cdh) or 1 (adh)");
      auto config = ::dark_hex::Config{
         spec_unsigned_as< size_t >(spec, GameFieldId::board_size),
         static_cast< ::dark_hex::RulesMode >(rules)};
      config.move_limit = spec_unsigned_as< size_t >(spec, GameFieldId::move_limit);
      config.validate();
      return env_type{std::move(config)};
   }
};

struct pursuit_evasion_game {
   using env_type = games::pursuit_evasion::Environment;
   static constexpr GameId id = GameId::pursuit_evasion;
   static constexpr std::string_view name = "pursuit_evasion";
   inline static constexpr std::array fields{FieldDescriptor{
      GameFieldId::rounds,
      "rounds",
      SpecKind::unsigned_integer,
      SpecValue{uint64_t{6}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{
         ::pursuit_evasion::Config{spec_unsigned_as< size_t >(spec, GameFieldId::rounds)}};
   }
};

struct oshi_zumo_game {
   using env_type = games::oshi_zumo::Environment;
   static constexpr GameId id = GameId::oshi_zumo;
   static constexpr std::string_view name = "oshi_zumo";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::size,
         "size",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{3}}},
      FieldDescriptor{
         GameFieldId::coins,
         "coins",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{50}}},
      FieldDescriptor{
         GameFieldId::min_bid,
         "min_bid",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{0}}},
      FieldDescriptor{
         GameFieldId::horizon,
         "horizon",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{9}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{::oshi_zumo::Config{
         spec_unsigned_as< size_t >(spec, GameFieldId::size),
         spec_unsigned_as< uint32_t >(spec, GameFieldId::coins),
         spec_unsigned_as< uint32_t >(spec, GameFieldId::min_bid),
         spec_unsigned_as< size_t >(spec, GameFieldId::horizon)}};
   }
};

struct shapley_game {
   using env_type = games::shapley::Environment;
   static constexpr GameId id = GameId::shapley;
   static constexpr std::string_view name = "shapley";
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct centipede_game {
   using env_type = games::centipede::Environment;
   static constexpr GameId id = GameId::centipede;
   static constexpr std::string_view name = "centipede";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::rounds,
         "rounds",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{4}}},
      FieldDescriptor{
         GameFieldId::pile_big,
         "pile_big",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{4}}},
      FieldDescriptor{
         GameFieldId::pile_small,
         "pile_small",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{1}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{::centipede::Config{
         spec_unsigned_as< size_t >(spec, GameFieldId::rounds),
         spec_unsigned_as< uint32_t >(spec, GameFieldId::pile_big),
         spec_unsigned_as< uint32_t >(spec, GameFieldId::pile_small)}};
   }
};

struct colonel_blotto_game {
   using env_type = games::colonel_blotto::Environment;
   static constexpr GameId id = GameId::colonel_blotto;
   static constexpr std::string_view name = "colonel_blotto";
   inline static constexpr std::array fields{FieldDescriptor{
      GameFieldId::budget,
      "budget",
      SpecKind::unsigned_integer,
      SpecValue{uint64_t{3}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{
         ::colonel_blotto::BlottoConfig{spec_unsigned_as< size_t >(spec, GameFieldId::budget)}};
   }
};

struct sheriff_game {
   using env_type = games::sheriff::Environment;
   static constexpr GameId id = GameId::sheriff;
   static constexpr std::string_view name = "sheriff";
   inline static constexpr std::array fields{
      FieldDescriptor{GameFieldId::v, "v", SpecKind::floating_point, SpecValue{5.}},
      FieldDescriptor{GameFieldId::p, "p", SpecKind::floating_point, SpecValue{1.}},
      FieldDescriptor{GameFieldId::s, "s", SpecKind::floating_point, SpecValue{1.}},
      FieldDescriptor{
         GameFieldId::n_max,
         "n_max",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{10}}},
      FieldDescriptor{
         GameFieldId::b_max,
         "b_max",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::rounds,
         "rounds",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{::sheriff::Config{
         spec_floating(spec, GameFieldId::v),
         spec_floating(spec, GameFieldId::p),
         spec_floating(spec, GameFieldId::s),
         spec_unsigned_as< size_t >(spec, GameFieldId::n_max),
         spec_unsigned_as< size_t >(spec, GameFieldId::b_max),
         spec_unsigned_as< size_t >(spec, GameFieldId::rounds)}};
   }
};

struct liars_dice_game {
   using env_type = games::liars_dice::Environment;
   static constexpr GameId id = GameId::liars_dice;
   static constexpr std::string_view name = "liars_dice";
   inline static constexpr std::array fields{
      FieldDescriptor{
         GameFieldId::n_players,
         "n_players",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{2}}},
      FieldDescriptor{
         GameFieldId::dice_per_player,
         "dice_per_player",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{1}}},
      FieldDescriptor{
         GameFieldId::n_faces,
         "n_faces",
         SpecKind::unsigned_integer,
         SpecValue{uint64_t{6}}}};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      auto config = ::liars_dice::DiceConfig{
         spec_unsigned_as< uint8_t >(spec, GameFieldId::n_players),
         spec_unsigned_as< uint8_t >(spec, GameFieldId::dice_per_player),
         spec_unsigned_as< uint8_t >(spec, GameFieldId::n_faces)};
      config.validate();
      return env_type{std::move(config)};
   }
};

using game_types = type_list<
   kuhn_game,
   leduc_game,
   rps_game,
   stratego_game,
   texas_holdem_game,
   goofspiel_game,
   three_player_goofspiel_game,
   battleship_game,
   battleship_gs_game,
   dark_hex_game,
   pursuit_evasion_game,
   oshi_zumo_game,
   shapley_game,
   centipede_game,
   colonel_blotto_game,
   sheriff_game,
   liars_dice_game >;

template < typename Tag >
[[nodiscard]] Result< GameHandle > make_game_impl(const GameSpec& input)
{
   auto normalized = normalized_spec< Tag >(input);
   if(not normalized)
      return std::unexpected(normalized.error());

   try {
      // Construct and initialize once during validation.  The session thunk repeats this from the
      // immutable spec, which gives every handle use an independent fresh environment/state.
      auto environment = Tag::make_env(*normalized);
      if(not environment)
         return std::unexpected(environment.error());
      (void) initial_world_state< Tag >(*environment);
      return GameHandle{std::move(*normalized)};
   } catch(const std::exception& exception) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to construct ") + std::string(Tag::name) + ": "
                    + exception.what(),
         .game = Tag::id});
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// solver profiles //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// A profile tag carries the exact factory config and the effective CFR/MCCFR config used for
// compile-time admissibility.  This is the only place where named profiles are declared.
struct vanilla_alternating_profile {
   static constexpr ProfileId id = ProfileId::vanilla_alternating;
   static constexpr SolverId solver = SolverId::vanilla_cfr;
   static constexpr std::string_view name = "vanilla_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRConfig{.update_mode = rm::UpdateMode::alternating};
   static constexpr auto effective_config = factory_config;
};

struct vanilla_simultaneous_profile {
   static constexpr ProfileId id = ProfileId::vanilla_simultaneous;
   static constexpr SolverId solver = SolverId::vanilla_cfr;
   static constexpr std::string_view name = "vanilla_cfr/simultaneous";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::simultaneous};
   static constexpr auto effective_config = factory_config;
};

struct cfr_plus_alternating_profile {
   static constexpr ProfileId id = ProfileId::cfr_plus_alternating;
   static constexpr SolverId solver = SolverId::cfr_plus;
   static constexpr std::string_view name = "cfr_plus/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRPlusConfig{
      .update_mode = rm::UpdateMode::alternating};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus};
};

struct lazy_alternating_profile {
   static constexpr ProfileId id = ProfileId::lazy_alternating;
   static constexpr SolverId solver = SolverId::lazy_cfr;
   static constexpr std::string_view name = "lazy_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRLazyConfig{
      .update_mode = rm::UpdateMode::alternating};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .lazy_update_mode = rm::CFRLazyUpdateMode::reach_threshold};
};

struct lazy_plus_alternating_profile {
   static constexpr ProfileId id = ProfileId::lazy_plus_alternating;
   static constexpr SolverId solver = SolverId::lazy_cfr_plus;
   static constexpr std::string_view name = "lazy_cfr_plus/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRLazyConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
      .lazy_update_mode = rm::CFRLazyUpdateMode::reach_threshold};
};

struct extragradient_alternating_profile {
   static constexpr ProfileId id = ProfileId::extragradient_alternating;
   static constexpr SolverId solver = SolverId::extragradient_cfr;
   static constexpr std::string_view name = "extragradient_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRExtragradientConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .regret_minimizing_mode = rm::RegretMinimizingMode::regret_matching_plus,
      .extragradient_mode = rm::CFRExtragradientMode::anchor_probe};
};

struct discounted_alternating_profile {
   static constexpr ProfileId id = ProfileId::discounted_alternating;
   static constexpr SolverId solver = SolverId::discounted_cfr;
   static constexpr std::string_view name = "discounted_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRDiscountedConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::discounted};
};

struct linear_alternating_profile {
   static constexpr ProfileId id = ProfileId::linear_alternating;
   static constexpr SolverId solver = SolverId::linear_cfr;
   static constexpr std::string_view name = "linear_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRLinearConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::linear};
};

struct exponential_alternating_profile {
   static constexpr ProfileId id = ProfileId::exponential_alternating;
   static constexpr SolverId solver = SolverId::exponential_cfr;
   static constexpr std::string_view name = "exponential_cfr/alternating";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRExponentialConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::exponential};
};

struct greedy_simultaneous_profile {
   static constexpr ProfileId id = ProfileId::greedy_simultaneous;
   static constexpr SolverId solver = SolverId::greedy_cfr;
   static constexpr std::string_view name = "greedy_cfr/simultaneous";
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRGreedyConfig{
      .update_mode = rm::UpdateMode::simultaneous};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::simultaneous,
      .weighting_mode = rm::CFRWeightingMode::greedy};
};

struct mccfr_outcome_lazy_profile {
   static constexpr ProfileId id = ProfileId::mccfr_outcome_lazy;
   static constexpr SolverId solver = SolverId::mccfr;
   static constexpr std::string_view name = "mccfr/outcome_sampling/lazy";
   static constexpr bool is_sampling = true;
   static constexpr auto factory_config = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy};
   static constexpr auto effective_config = factory_config;
};

struct mccfr_chance_sampling_profile {
   static constexpr ProfileId id = ProfileId::mccfr_chance_sampling;
   static constexpr SolverId solver = SolverId::mccfr;
   static constexpr std::string_view name = "mccfr/chance_sampling";
   static constexpr bool is_sampling = true;
   static constexpr auto factory_config = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::chance_sampling,
      .weighting = rm::MCCFRWeightingMode::none};
   static constexpr auto effective_config = factory_config;
};

struct mccfr_external_sampling_profile {
   static constexpr ProfileId id = ProfileId::mccfr_external_sampling;
   static constexpr SolverId solver = SolverId::mccfr;
   static constexpr std::string_view name = "mccfr/external_sampling";
   static constexpr bool is_sampling = true;
   static constexpr auto factory_config = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::external_sampling,
      .weighting = rm::MCCFRWeightingMode::stochastic};
   static constexpr auto effective_config = factory_config;
};

struct mccfr_pure_cfr_profile {
   static constexpr ProfileId id = ProfileId::mccfr_pure_cfr;
   static constexpr SolverId solver = SolverId::mccfr;
   static constexpr std::string_view name = "mccfr/pure_cfr";
   static constexpr bool is_sampling = true;
   static constexpr auto factory_config = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::pure_cfr,
      .weighting = rm::MCCFRWeightingMode::none};
   static constexpr auto effective_config = factory_config;
};

struct mccfr_plus_alternating_profile {
   static constexpr ProfileId id = ProfileId::mccfr_plus_alternating;
   static constexpr SolverId solver = SolverId::mccfr_plus;
   static constexpr std::string_view name = "mccfr_plus/outcome_sampling/lazy";
   static constexpr bool is_sampling = true;
   static constexpr auto factory_config = rm::MCCFRPlusConfig{};
   static constexpr auto effective_config = rm::MCCFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .algorithm = rm::MCCFRAlgorithmMode::outcome_sampling,
      .weighting = rm::MCCFRWeightingMode::lazy,
      .regret_minimizing_mode = rm::RegretMinimizingMode::predictive_regret_matching_plus};
};

using profile_types = type_list<
   vanilla_alternating_profile,
   vanilla_simultaneous_profile,
   cfr_plus_alternating_profile,
   lazy_alternating_profile,
   lazy_plus_alternating_profile,
   extragradient_alternating_profile,
   discounted_alternating_profile,
   linear_alternating_profile,
   exponential_alternating_profile,
   greedy_simultaneous_profile,
   mccfr_outcome_lazy_profile,
   mccfr_chance_sampling_profile,
   mccfr_external_sampling_profile,
   mccfr_pure_cfr_profile,
   mccfr_plus_alternating_profile >;

template < auto Config >
[[nodiscard]] consteval bool mccfr_config_supported()
{
   constexpr auto config = Config;
   constexpr auto variance_reduction = rm::effective_variance_reduction(config);
   if constexpr(config.pruning_mode == rm::CFRPruningMode::regret_based or config.pruning_mode == rm::CFRPruningMode::dynamic_thresholding or (config.pruning_mode == rm::CFRPruningMode::partial and config.algorithm != rm::MCCFRAlgorithmMode::chance_sampling)) {
      return false;
   }
   if constexpr(config.algorithm == rm::MCCFRAlgorithmMode::external_sampling and (config.update_mode != rm::UpdateMode::alternating or config.weighting != rm::MCCFRWeightingMode::stochastic)) {
      return false;
   }
   if constexpr(variance_reduction != rm::VarianceReductionMode::none and config.algorithm != rm::MCCFRAlgorithmMode::outcome_sampling) {
      return false;
   }
   if constexpr(variance_reduction != rm::VarianceReductionMode::none and config.update_mode != rm::UpdateMode::alternating) {
      return false;
   }
   if constexpr(config.updater_sampling == rm::UpdaterSamplingMode::fixed_uniform and variance_reduction != rm::VarianceReductionMode::history_value) {
      return false;
   }
   if constexpr(config.baseline_update_rule == rm::BaselineUpdateRule::predictive and variance_reduction == rm::VarianceReductionMode::none) {
      return false;
   }
   if constexpr(
      not rm::detail::mccfr_admissible_rm_mode< config.regret_minimizing_mode >
      or not rm::detail::mccfr_rm_mode_compatible<
         config.algorithm,
         config.regret_minimizing_mode >
   ) {
      return false;
   }
   return true;
}

template < typename Env >
[[nodiscard]] consteval bool tabular_environment_supported()
{
   if constexpr(not concepts::fosg< Env >) {
      return false;
   } else {
      using action_type = auto_action_type< Env >;
      using info_state_type = auto_info_state_type< Env >;
      using action_policy = HashmapActionPolicy< action_type >;
      using policy = TabularPolicy< info_state_type, action_policy >;
      using default_policy = UniformPolicy< info_state_type, action_policy >;
      return concepts::
         tabular_cfr_requirements< Env, policy, policy, default_policy, default_policy >;
   }
}

template < typename Env, typename Profile >
[[nodiscard]] consteval bool profile_supported()
{
   if constexpr(not tabular_environment_supported< Env >()) {
      return false;
   } else if constexpr(Profile::is_sampling) {
      return mccfr_config_supported< Profile::effective_config >();
   } else {
      return rm::detail::sanity_check_cfr_config< Profile::effective_config >();
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// session adapter //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename PolicyMap >
[[nodiscard]] size_t policy_entry_count(const PolicyMap& policies)
{
   size_t count = 0;
   for(const auto& [player, state_policy] : policies) {
      (void) player;
      count += static_cast< size_t >(state_policy.size());
   }
   return count;
}

template < typename PolicyMap >
void append_policy_entries(const PolicyMap& policies, PolicyView& output)
{
   for(const auto& [player, state_policy] : policies) {
      for(const auto& [info_state, action_policy] : state_policy) {
         (void) info_state;
         PolicyEntry entry{.player = player, .action_probabilities = {}};
         entry.action_probabilities.reserve(action_policy.size());
         for(const auto& [action, probability] : action_policy) {
            (void) action;
            entry.action_probabilities.emplace_back(probability);
         }
         output.entries.emplace_back(std::move(entry));
      }
   }

   // The concrete table is intentionally optimized for solver traversal and is not required to
   // expose a stable iteration order. The temporary ABI does not expose the infoset key itself,
   // so sorting the value snapshot is sufficient to make the observable view deterministic even
   // when the solver's outer table is hash-backed. Equal snapshots are indistinguishable at this
   // ABI boundary.
   std::ranges::sort(output.entries, [](const PolicyEntry& left, const PolicyEntry& right) {
      if(left.player != right.player) {
         return static_cast< int >(left.player) < static_cast< int >(right.player);
      }
      return std::lexicographical_compare(
         left.action_probabilities.begin(),
         left.action_probabilities.end(),
         right.action_probabilities.begin(),
         right.action_probabilities.end()
      );
   });
   for(size_t ordinal = 0; ordinal < output.entries.size(); ++ordinal) {
      output.entries[ordinal].info_state_ordinal = ordinal;
   }
}

template < typename Solver, GameId Game, SolverId SolverFamily, ProfileId Profile >
struct SessionModel {
   Solver solver;

   explicit SessionModel(Solver concrete_solver) : solver(std::move(concrete_solver)) {}

   static void destroy(void* object) noexcept { delete static_cast< SessionModel* >(object); }

   template < typename T >
   [[nodiscard]] static Result< T > operation_error(CapabilityErrorCode code, std::string message)
   {
      return std::unexpected(CapabilityError{
         .code = code,
         .message = std::move(message),
         .game = Game,
         .solver = SolverFamily,
         .profile = Profile});
   }

   static Result< IterateResult > iterate(void* object, size_t iterations)
   {
      auto& self = *static_cast< SessionModel* >(object);
      try {
         const size_t first_iteration = self.solver.iteration();
         auto roots = self.solver.iterate(iterations);
         IterateResult result{
            .first_iteration = first_iteration,
            .last_iteration = self.solver.iteration(),
            .iterations = {}};
         result.iterations.reserve(roots.size());
         size_t index = 0;
         for(const auto& root : roots) {
            IterationResult iteration{.iteration = first_iteration + index++, .root_values = {}};
            for(const auto [player, value] : root.get()) {
               iteration.root_values.push_back(RootValue{.player = player, .value = value});
            }
            result.iterations.emplace_back(std::move(iteration));
         }
         return result;
      } catch(const std::exception& exception) {
         return operation_error< IterateResult >(
            CapabilityErrorCode::session_failure,
            std::string("solver iteration failed: ") + exception.what()
         );
      }
   }

   // Temporary adapter: until the sibling solver primitive lands, advance has the same coarse
   // semantics as iterate.  It is deliberately isolated to this one vtable member.
   static Result< IterateResult > advance(void* object, size_t iterations)
   {
      return iterate(object, iterations);
   }

   static Result< TraceResult > trace(const void*, const TraceRequest&)
   {
      return operation_error< TraceResult >(
         CapabilityErrorCode::operation_unavailable,
         "trace awaits the solver trace primitive; no per-node callback is installed"
      );
   }

   static Result< SessionStats > stats(const void* object)
   {
      const auto& self = *static_cast< const SessionModel* >(object);
      try {
         const auto& current = self.solver.policy();
         auto&& average = self.solver.average_policy();
         return SessionStats{
            .game = Game,
            .solver = SolverFamily,
            .profile = Profile,
            .iteration = self.solver.iteration(),
            .cycle = self.solver.cycle(),
            .player_count = self.solver.env().players(self.solver.root_state()).size(),
            .current_policy_entries = policy_entry_count(current),
            .average_policy_entries = policy_entry_count(average)};
      } catch(const std::exception& exception) {
         return operation_error< SessionStats >(
            CapabilityErrorCode::session_failure,
            std::string("solver statistics failed: ") + exception.what()
         );
      }
   }

   static Result< PolicyView > policy_view(const void* object, PolicyViewKind kind)
   {
      const auto& self = *static_cast< const SessionModel* >(object);
      try {
         PolicyView output{
            .kind = kind, .complete = true, .temporary_adapter = true, .entries = {}};
         if(kind == PolicyViewKind::current) {
            append_policy_entries(self.solver.policy(), output);
         } else {
            auto&& average = self.solver.average_policy();
            append_policy_entries(average, output);
         }
         return output;
      } catch(const std::exception& exception) {
         return operation_error< PolicyView >(
            CapabilityErrorCode::session_failure,
            std::string("policy view failed: ") + exception.what()
         );
      }
   }

   inline static constexpr SolverSessionOps ops{
      .destroy = &destroy,
      .iterate = &iterate,
      .advance = &advance,
      .trace = &trace,
      .stats = &stats,
      .policy_view = &policy_view};
};

template < typename Profile, typename Env >
[[nodiscard]] auto make_concrete_solver(
   Env environment,
   uptr< auto_world_state_type< Env > > root,
   SessionOptions options
)
{
   using action_type = auto_action_type< Env >;
   using info_state_type = auto_info_state_type< Env >;
   using policy_type = TabularPolicy< info_state_type, HashmapActionPolicy< action_type > >;

   policy_type current_policy;
   policy_type average_policy;
   if constexpr(Profile::solver == SolverId::lazy_cfr_plus) {
      // CFRLazyConfig is shared by LazyCFR and LazyCFRPlus.  The generic factory
      // selects LazyCFR by config type, so this one family-specific entry point
      // is selected explicitly from the profile's stable solver ID.
      return factory::make_cfr_lazy_plus< Profile::factory_config, true >(
         std::move(environment),
         std::move(root),
         std::move(current_policy),
         std::move(average_policy)
      );
   } else if constexpr(Profile::is_sampling) {
      return factory::make_cfr< Profile::factory_config, true >(
         std::move(environment),
         std::move(root),
         std::move(current_policy),
         std::move(average_policy),
         options.epsilon,
         options.seed
      );
   } else {
      return factory::make_cfr< Profile::factory_config, true >(
         std::move(environment),
         std::move(root),
         std::move(current_policy),
         std::move(average_policy)
      );
   }
}

template < typename Game, typename Profile >
[[nodiscard]] Result< SolverSession >
make_session_impl(const GameHandle& handle, SessionOptions options)
{
   if(handle.game_id() != Game::id) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "GameHandle ID does not match its selected capability thunk",
         .game = handle.game_id(),
         .solver = Profile::solver,
         .profile = Profile::id});
   }

   try {
      auto normalized = normalized_spec< Game >(handle.spec());
      if(not normalized) {
         auto error = normalized.error();
         error.code = CapabilityErrorCode::invalid_handle;
         error.message = "GameHandle carries an invalid static GameSpec: " + error.message;
         error.game = Game::id;
         error.solver = Profile::solver;
         error.profile = Profile::id;
         return std::unexpected(std::move(error));
      }
      auto environment_result = Game::make_env(*normalized);
      if(not environment_result)
         return std::unexpected(environment_result.error());
      auto environment = std::move(*environment_result);
      auto root_state = initial_world_state< Game >(environment);
      auto root = std::make_unique< auto_world_state_type< typename Game::env_type > >(
         std::move(root_state)
      );
      auto solver = make_concrete_solver< Profile >(
         std::move(environment), std::move(root), options
      );
      using solver_type = decltype(solver);
      using model_type = SessionModel< solver_type, Game::id, Profile::solver, Profile::id >;
      return SolverSession::make< model_type >(std::move(solver));
   } catch(const std::exception& exception) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to create ") + std::string(Profile::name) + " on "
                    + std::string(Game::name) + ": " + exception.what(),
         .game = Game::id,
         .solver = Profile::solver,
         .profile = Profile::id});
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// consteval catalogs ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

template < typename Game >
[[nodiscard]] consteval GameDescriptor make_game_descriptor()
{
   constexpr size_t min_players = default_min_players< typename Game::env_type >();
   constexpr size_t max_players = default_max_players< typename Game::env_type >();
   return GameDescriptor{
      .id = Game::id,
      .name = Game::name,
      .min_players = min_players,
      .max_players = max_players,
      .stochasticity = Game::env_type::stochasticity(),
      .fields = std::span< const FieldDescriptor >{Game::fields},
      .create = &make_game_impl< Game >};
}

template < typename... Games >
[[nodiscard]] consteval auto make_game_descriptors(type_list< Games... >)
{
   static_assert(unique_values(std::array{Games::id...}), "static game IDs must be unique");
   static_assert(
      (unique_field_ids(Games::fields) and ...),
      "fields within a static game specification must be unique"
   );
   return std::array{make_game_descriptor< Games >()...};
}

template < typename Profile >
[[nodiscard]] consteval ProfileDescriptor make_profile_descriptor()
{
   return ProfileDescriptor{.id = Profile::id, .solver = Profile::solver, .name = Profile::name};
}

template < typename... Profiles >
[[nodiscard]] consteval auto make_profile_descriptors(type_list< Profiles... >)
{
   static_assert(unique_values(std::array{Profiles::id...}), "static profile IDs must be unique");
   return std::array{make_profile_descriptor< Profiles >()...};
}

inline constexpr std::array solver_descriptors{
   SolverDescriptor{.id = SolverId::vanilla_cfr, .name = "vanilla_cfr"},
   SolverDescriptor{.id = SolverId::cfr_plus, .name = "cfr_plus"},
   SolverDescriptor{.id = SolverId::lazy_cfr, .name = "lazy_cfr"},
   SolverDescriptor{.id = SolverId::lazy_cfr_plus, .name = "lazy_cfr_plus"},
   SolverDescriptor{.id = SolverId::extragradient_cfr, .name = "extragradient_cfr"},
   SolverDescriptor{.id = SolverId::discounted_cfr, .name = "discounted_cfr"},
   SolverDescriptor{.id = SolverId::linear_cfr, .name = "linear_cfr"},
   SolverDescriptor{.id = SolverId::exponential_cfr, .name = "exponential_cfr"},
   SolverDescriptor{.id = SolverId::greedy_cfr, .name = "greedy_cfr"},
   SolverDescriptor{.id = SolverId::mccfr, .name = "mccfr"},
   SolverDescriptor{.id = SolverId::mccfr_plus, .name = "mccfr_plus"}};

template < typename Game, typename... Profiles >
[[nodiscard]] consteval size_t capability_count_for(type_list< Profiles... >)
{
   return (static_cast< size_t >(profile_supported< typename Game::env_type, Profiles >()) + ...);
}

template < typename... Games, typename... Profiles >
[[nodiscard]] consteval size_t
capability_count(type_list< Games... > games, type_list< Profiles... > profiles)
{
   (void) games;
   (void) profiles;
   return (capability_count_for< Games >(type_list< Profiles... >{}) + ...);
}

template < typename Game, typename Profile, size_t Count >
constexpr void append_capability(std::array< CapabilityDescriptor, Count >& output, size_t& index)
{
   if constexpr(profile_supported< typename Game::env_type, Profile >()) {
      output[index++] = CapabilityDescriptor{
         .game = Game::id,
         .solver = Profile::solver,
         .profile = Profile::id,
         .create = &make_session_impl< Game, Profile >};
   }
}

template < typename Game, typename... Profiles, size_t Count >
constexpr void append_game_capabilities(
   std::array< CapabilityDescriptor, Count >& output,
   size_t& index,
   type_list< Profiles... > profiles
)
{
   (void) profiles;
   (append_capability< Game, Profiles >(output, index), ...);
}

template < size_t Count, typename... Games, typename... Profiles >
[[nodiscard]] consteval auto
make_capabilities(type_list< Games... > games, type_list< Profiles... > profiles)
{
   (void) games;
   (void) profiles;
   std::array< CapabilityDescriptor, Count > output{};
   size_t index = 0;
   (append_game_capabilities< Games >(output, index, profiles), ...);
   return output;
}

template < size_t Count >
[[nodiscard]] consteval bool unique_capabilities(
   const std::array< CapabilityDescriptor, Count >& capabilities
)
{
   for(size_t left = 0; left < Count; ++left) {
      if(capabilities[left].create == nullptr)
         return false;
      for(size_t right = left + 1; right < Count; ++right) {
         if(capabilities[left].game == capabilities[right].game
            and capabilities[left].solver == capabilities[right].solver
            and capabilities[left].profile == capabilities[right].profile) {
            return false;
         }
      }
   }
   return true;
}

inline constexpr auto game_descriptors = make_game_descriptors(game_types{});
inline constexpr auto profile_descriptors = make_profile_descriptors(profile_types{});
inline constexpr size_t capability_count_v = capability_count(game_types{}, profile_types{});

// The helper above is intentionally specialized through this lambda so the array extent remains a
// compile-time value while the game/profile type-lists stay the only cross-product declaration.
inline constexpr auto all_capability_descriptors = [] {
   return make_capabilities< capability_count_v >(game_types{}, profile_types{});
}();

static_assert(
   unique_capabilities(all_capability_descriptors),
   "static capability entries must be complete and unique"
);

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// public lookup API ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

[[nodiscard]] inline const StaticCatalog& catalog() noexcept
{
   static constexpr StaticCatalog value{
      .games = detail::game_descriptors,
      .solvers = detail::solver_descriptors,
      .profiles = detail::profile_descriptors,
      .combinations = detail::all_capability_descriptors};
   return value;
}

[[nodiscard]] inline std::span< const GameDescriptor > games() noexcept
{
   return catalog().games;
}

[[nodiscard]] inline std::span< const SolverDescriptor > solvers() noexcept
{
   return catalog().solvers;
}

[[nodiscard]] inline std::span< const ProfileDescriptor > profiles() noexcept
{
   return catalog().profiles;
}

[[nodiscard]] inline std::span< const CapabilityDescriptor > capabilities() noexcept
{
   return catalog().combinations;
}

[[nodiscard]] inline std::vector< CapabilityDescriptor > capabilities_for(GameId game)
{
   std::vector< CapabilityDescriptor > result;
   for(const auto& capability : catalog().combinations) {
      if(capability.game == game)
         result.emplace_back(capability);
   }
   return result;
}

[[nodiscard]] inline std::vector< ProfileDescriptor > profiles_for(SolverId solver)
{
   std::vector< ProfileDescriptor > result;
   for(const auto& profile : catalog().profiles) {
      if(profile.solver == solver)
         result.emplace_back(profile);
   }
   return result;
}

[[nodiscard]] inline const GameDescriptor* find_game(GameId id) noexcept
{
   for(const auto& descriptor : catalog().games) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

[[nodiscard]] inline const SolverDescriptor* find_solver(SolverId id) noexcept
{
   for(const auto& descriptor : catalog().solvers) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

[[nodiscard]] inline const ProfileDescriptor* find_profile(ProfileId id) noexcept
{
   for(const auto& descriptor : catalog().profiles) {
      if(descriptor.id == id)
         return &descriptor;
   }
   return nullptr;
}

[[nodiscard]] inline const CapabilityDescriptor*
find_capability(GameId game, SolverId solver, ProfileId profile) noexcept
{
   for(const auto& descriptor : catalog().combinations) {
      if(descriptor.game == game && descriptor.solver == solver && descriptor.profile == profile) {
         return &descriptor;
      }
   }
   return nullptr;
}

[[nodiscard]] inline GameSpec GameSpec::defaults(GameId id)
{
   GameSpec spec{id};
   if(const auto* descriptor = find_game(id); descriptor != nullptr) {
      for(const auto& field : descriptor->fields)
         spec.set(field.id, field.default_value);
   }
   return spec;
}

[[nodiscard]] inline Result< GameHandle > make_game(const GameSpec& spec)
{
   const auto* descriptor = find_game(spec.game_id());
   if(descriptor == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_game,
         .message = "GameSpec refers to an unknown static game ID",
         .game = spec.game_id()});
   }
   return descriptor->create(spec);
}

[[nodiscard]] inline Result< SolverSession >
make_session(const GameHandle& handle, SolverId solver, ProfileId profile, SessionOptions options)
{
   if(find_game(handle.game_id()) == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "GameHandle refers to an unknown static game ID",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(find_solver(solver) == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_solver,
         .message = "solver ID is not present in the immutable solver catalog",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   const auto* profile_descriptor = find_profile(profile);
   if(profile_descriptor == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unknown_profile,
         .message = "profile ID is not present in the immutable profile catalog",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(profile_descriptor->solver != solver) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::profile_solver_mismatch,
         .message = "profile belongs to a different solver family",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "sampling epsilon must be finite and in [0, 1]",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   const auto* capability = find_capability(handle.game_id(), solver, profile);
   if(capability == nullptr) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::unsupported_combination,
         .message = "game/solver/profile combination is not accepted by the static capability "
                    "matrix",
         .game = handle.game_id(),
         .solver = solver,
         .profile = profile});
   }
   return capability->create(handle, options);
}

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_CATALOG_HPP
