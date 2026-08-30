#ifndef NOR_BINDING_RUNTIME_CATALOG_HPP
#define NOR_BINDING_RUNTIME_CATALOG_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
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

template < typename Game >
[[nodiscard]] consteval std::string_view game_name();

template < typename Profile >
[[nodiscard]] consteval std::string_view profile_name();

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

template < typename T, typename... Ts >
inline constexpr bool contains_type_v = (std::is_same_v< T, Ts > or ...);

/// Whether @p Needle is one of the types in a type list.
template < typename Needle, typename... Ts >
[[nodiscard]] consteval bool list_contains(type_list< Ts... >)
{
   return contains_type_v< Needle, Ts... >;
}

/// Whether two type lists name exactly the same set of types, in any order.
template < typename... Left, typename... Right >
[[nodiscard]] consteval bool same_type_set(type_list< Left... >, type_list< Right... >)
{
   return sizeof...(Left) == sizeof...(Right) and (contains_type_v< Left, Right... > and ...)
          and (contains_type_v< Right, Left... > and ...);
}

template < typename... Games >
[[nodiscard]] consteval bool unique_game_ids(type_list< Games... >)
{
   return unique_values(std::array{Games::id...});
}

template < typename... Games >
[[nodiscard]] consteval bool unique_game_field_ids(type_list< Games... >)
{
   return (unique_field_ids(Games::fields) and ...);
}

template < typename... Profiles >
[[nodiscard]] consteval bool unique_profile_ids(type_list< Profiles... >)
{
   return unique_values(std::array{Profiles::id...});
}

template < typename... Games >
[[nodiscard]] consteval size_t type_list_size(type_list< Games... >)
{
   return sizeof...(Games);
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
            "field is not accepted by game '" + std::string(game_name< Tag >()) + "'."
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
// richer environment types expose initial_world_state(). A tag may also build the root itself
// from its normalized spec, which is what games whose root is a board position rather than an
// environment property need. Keep that difference inside the binding-runtime adapter so neither
// the registry nor libnor needs a special-case API.
template < typename Game >
[[nodiscard]] auto
initial_world_state(const typename Game::env_type& environment, const GameSpec& spec)
{
   using env_type = typename Game::env_type;
   if constexpr(requires(const GameSpec& s) { Game::world_state_for(s); }) {
      return Game::world_state_for(spec);
   } else if constexpr(requires(const env_type& env) { env.initial_world_state(); }) {
      return environment.initial_world_state();
   } else {
      return typename env_type::world_state_type{};
   }
}

template < typename... Args >
[[nodiscard]] consteval FieldDescriptor
reflected_field(GameFieldId id, SpecKind kind, Args&&... args)
{
   return FieldDescriptor{
      .id = id,
      .name = meta::enum_name(id),
      .kind = kind,
      .default_value = SpecValue{std::forward< Args >(args)...}};
}

template < typename Game >
[[nodiscard]] consteval std::string_view game_name()
{
   return meta::enum_name(Game::id);
}

template < typename Profile >
[[nodiscard]] consteval std::string_view profile_name()
{
   return meta::enum_name(Profile::id);
}

// Each tag is a single source of truth for a stable ID, its fields, and its concrete environment
// constructor.  The arrays below are intentionally explicit: adding a game requires adding one
// tag here, after which descriptor and capability generation remains generic.
struct kuhn_game {
   using env_type = games::kuhn::Environment;
   static constexpr GameId id = GameId::kuhn;
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct leduc_game {
   using env_type = games::leduc::Environment;
   static constexpr GameId id = GameId::leduc;
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct rps_game {
   using env_type = games::rps::Environment;
   static constexpr GameId id = GameId::rps;
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct stratego_game {
   using env_type = games::stratego::Environment;
   static constexpr GameId id = GameId::stratego;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::board_size, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::max_turn_count, SpecKind::unsigned_integer, uint64_t{10})};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }

   /**
    * @brief The 3x3 flag/spy opening that the Stratego implementation's own suite plays out.
    *
    * Stratego's convenience constructor derives a start field from a helper that does not produce
    * a playable position, and an empty setup default-constructs but has no legal first move. The
    * board below is the same explicit, fully placed position the game's `StrategoState3x3` fixture
    * uses, so the registered default is a real game rather than a state that only survives until
    * the first traversal:
    *
    * ```
    * ---------------- BLUE holds the flag at (0,0) and a spy at (0,1);
    * |    | 1R | 0R | RED holds a spy at (2,1) and the flag at (2,2).
    * ----------------
    * |    |    |    |
    * ----------------
    * | 0B | 1B |    |
    * ----------------
    * ```
    *
    * The turn cap keeps the default catalog entry a tractable tree; a caller that wants a longer
    * or larger game supplies the spec fields.
    */
   static env_type::world_state_type world_state_for(const GameSpec& spec)
   {
      using ::stratego::Config;
      using ::stratego::Position2D;
      using ::stratego::Team;
      using ::stratego::Token;

      const auto board_size = spec.contains(GameFieldId::board_size)
                                 ? spec_unsigned_as< size_t >(spec, GameFieldId::board_size)
                                 : size_t{3};
      const auto max_turn_count = spec.contains(GameFieldId::max_turn_count)
                                     ? spec_unsigned_as< size_t >(spec, GameFieldId::max_turn_count)
                                     : size_t{10};
      if(board_size < 3) {
         throw std::invalid_argument("stratego board_size must be at least 3");
      }
      if(max_turn_count == 0) {
         throw std::invalid_argument("stratego max_turn_count must be greater than zero");
      }

      // Both teams get a flag and a spy on opposite corners of the board, which is a legal
      // position for every square board of size three or more.
      const auto last = board_size - 1;
      Config::setup_t blue{{Position2D{0, 0}, Token::flag}, {Position2D{0, 1}, Token::spy}};
      Config::setup_t red{
         {Position2D{static_cast< int >(last), static_cast< int >(last) - 1}, Token::spy},
         {Position2D{static_cast< int >(last), static_cast< int >(last)}, Token::flag}};

      Config config{
         Team::BLUE,
         board_size,
         std::map< Team, std::optional< Config::setup_t > >{
            {Team::BLUE, std::make_optional(std::move(blue))},
            {Team::RED, std::make_optional(std::move(red))}},
         std::vector< Position2D >{},
         true,
         true,
         max_turn_count};
      return env_type::world_state_type{std::move(config), size_t{0}};
   }
};

struct texas_holdem_game {
   using env_type = games::texholdem::Environment;
   static constexpr GameId id = GameId::texas_holdem;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::n_players, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::starting_stack, SpecKind::floating_point, 200.),
      reflected_field(GameFieldId::small_blind, SpecKind::floating_point, 1.),
      reflected_field(GameFieldId::big_blind, SpecKind::floating_point, 2.),
      reflected_field(GameFieldId::deck_size, SpecKind::unsigned_integer, uint64_t{52})};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      auto config = ::texholdem::PokerConfig{
         spec_unsigned_as< size_t >(spec, GameFieldId::n_players),
         spec_floating(spec, GameFieldId::starting_stack),
         spec_floating(spec, GameFieldId::small_blind),
         spec_floating(spec, GameFieldId::big_blind)};
      config.deck_size = spec_unsigned_as< size_t >(spec, GameFieldId::deck_size);
      config.validate();
      return env_type{std::move(config)};
   }
};

struct goofspiel_game {
   using env_type = games::goofspiel::Environment;
   static constexpr GameId id = GameId::goofspiel;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::deck_size, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::imp_info, SpecKind::boolean, false)};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::deck_size, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::imp_info, SpecKind::boolean, false),
      reflected_field(GameFieldId::split_half_deal, SpecKind::boolean, false)};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::rows, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::cols, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::ships_per_fleet, SpecKind::unsigned_integer, uint64_t{1}),
      reflected_field(GameFieldId::max_shots, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::ship_value, SpecKind::floating_point, 2.)};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::rows, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::cols, SpecKind::unsigned_integer, uint64_t{1}),
      reflected_field(GameFieldId::max_shots, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::loss_multiplier, SpecKind::floating_point, 2.)};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::board_size, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::rules_mode, SpecKind::unsigned_integer, uint64_t{0}),
      reflected_field(GameFieldId::move_limit, SpecKind::unsigned_integer, uint64_t{0})};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::rounds, SpecKind::unsigned_integer, uint64_t{6})};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{
         ::pursuit_evasion::Config{spec_unsigned_as< size_t >(spec, GameFieldId::rounds)}};
   }
};

struct oshi_zumo_game {
   using env_type = games::oshi_zumo::Environment;
   static constexpr GameId id = GameId::oshi_zumo;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::size, SpecKind::unsigned_integer, uint64_t{3}),
      reflected_field(GameFieldId::coins, SpecKind::unsigned_integer, uint64_t{50}),
      reflected_field(GameFieldId::min_bid, SpecKind::unsigned_integer, uint64_t{0}),
      reflected_field(GameFieldId::horizon, SpecKind::unsigned_integer, uint64_t{9})};

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
   inline static constexpr std::array< FieldDescriptor, 0 > fields{};

   static Result< env_type > make_env(const GameSpec&) { return env_type{}; }
};

struct centipede_game {
   using env_type = games::centipede::Environment;
   static constexpr GameId id = GameId::centipede;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::rounds, SpecKind::unsigned_integer, uint64_t{4}),
      reflected_field(GameFieldId::pile_big, SpecKind::unsigned_integer, uint64_t{4}),
      reflected_field(GameFieldId::pile_small, SpecKind::unsigned_integer, uint64_t{1})};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::budget, SpecKind::unsigned_integer, uint64_t{3})};

   static Result< env_type > make_env(const GameSpec& spec)
   {
      return env_type{
         ::colonel_blotto::BlottoConfig{spec_unsigned_as< size_t >(spec, GameFieldId::budget)}};
   }
};

struct sheriff_game {
   using env_type = games::sheriff::Environment;
   static constexpr GameId id = GameId::sheriff;
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::v, SpecKind::floating_point, 5.),
      reflected_field(GameFieldId::p, SpecKind::floating_point, 1.),
      reflected_field(GameFieldId::s, SpecKind::floating_point, 1.),
      reflected_field(GameFieldId::n_max, SpecKind::unsigned_integer, uint64_t{10}),
      reflected_field(GameFieldId::b_max, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::rounds, SpecKind::unsigned_integer, uint64_t{2})};

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
   inline static constexpr std::array fields{
      reflected_field(GameFieldId::n_players, SpecKind::unsigned_integer, uint64_t{2}),
      reflected_field(GameFieldId::dice_per_player, SpecKind::unsigned_integer, uint64_t{1}),
      reflected_field(GameFieldId::n_faces, SpecKind::unsigned_integer, uint64_t{6})};

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
      (void) initial_world_state< Tag >(*environment, *normalized);
      return GameHandle{std::move(*normalized)};
   } catch(const std::exception& exception) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to construct ") + std::string(game_name< Tag >()) + ": "
                    + exception.what(),
         .game = Tag::id});
   } catch(...) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to construct ") + std::string(game_name< Tag >())
                    + ": the environment threw a non-standard exception",
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
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRConfig{.update_mode = rm::UpdateMode::alternating};
   static constexpr auto effective_config = factory_config;
};

struct vanilla_simultaneous_profile {
   static constexpr ProfileId id = ProfileId::vanilla_simultaneous;
   static constexpr SolverId solver = SolverId::vanilla_cfr;
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::simultaneous};
   static constexpr auto effective_config = factory_config;
};

struct cfr_plus_alternating_profile {
   static constexpr ProfileId id = ProfileId::cfr_plus_alternating;
   static constexpr SolverId solver = SolverId::cfr_plus;
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
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRDiscountedConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::discounted};
};

struct linear_alternating_profile {
   static constexpr ProfileId id = ProfileId::linear_alternating;
   static constexpr SolverId solver = SolverId::linear_cfr;
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRLinearConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::linear};
};

struct exponential_alternating_profile {
   static constexpr ProfileId id = ProfileId::exponential_alternating;
   static constexpr SolverId solver = SolverId::exponential_cfr;
   static constexpr bool is_sampling = false;
   static constexpr auto factory_config = rm::CFRExponentialConfig{};
   static constexpr auto effective_config = rm::CFRConfig{
      .update_mode = rm::UpdateMode::alternating,
      .weighting_mode = rm::CFRWeightingMode::exponential};
};

struct greedy_simultaneous_profile {
   static constexpr ProfileId id = ProfileId::greedy_simultaneous;
   static constexpr SolverId solver = SolverId::greedy_cfr;
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
[[nodiscard]] consteval bool factory_constructible()
{
   if constexpr(not tabular_environment_supported< Env >()) {
      return false;
   } else {
      using action_type = auto_action_type< Env >;
      using info_state_type = auto_info_state_type< Env >;
      using world_state_type = auto_world_state_type< Env >;
      using policy_type = TabularPolicy< info_state_type, HashmapActionPolicy< action_type > >;

      // Check the actual factory expression, including the exact root and policy types used by
      // the runtime thunk. This keeps the capability matrix honest when a solver family changes
      // its constructor requirements; a Cartesian count is never used as validation.
      if constexpr(Profile::solver == SolverId::lazy_cfr_plus) {
         return requires(
            Env environment, uptr< world_state_type > root, policy_type current, policy_type average
         ) {
            factory::make_cfr_lazy_plus< Profile::factory_config, true >(
               std::move(environment), std::move(root), std::move(current), std::move(average)
            );
         };
      } else if constexpr(Profile::is_sampling) {
         return requires(
            Env environment, uptr< world_state_type > root, policy_type current, policy_type average
         ) {
            factory::make_cfr< Profile::factory_config, true >(
               std::move(environment),
               std::move(root),
               std::move(current),
               std::move(average),
               double{},
               size_t{}
            );
         };
      } else {
         return requires(
            Env environment, uptr< world_state_type > root, policy_type current, policy_type average
         ) {
            factory::make_cfr< Profile::factory_config, true >(
               std::move(environment), std::move(root), std::move(current), std::move(average)
            );
         };
      }
   }
}

template < typename Env, typename Profile >
[[nodiscard]] consteval bool profile_supported()
{
   if constexpr(not tabular_environment_supported< Env >()) {
      return false;
   } else if constexpr(Profile::is_sampling) {
      return mccfr_config_supported< Profile::effective_config >()
             and factory_constructible< Env, Profile >();
   } else {
      return rm::detail::sanity_check_cfr_config< Profile::effective_config >()
             and factory_constructible< Env, Profile >();
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// session adapter //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// This is the sole implementation-side constructor for the public move-only session handle. It
// is declared only as a friend in types.hpp so a binding consumer cannot manufacture a session
// with an arbitrary model/vtable and thereby bypass the capability registry.
struct SessionFactory {
   template < typename Model, typename... Args >
   [[nodiscard]] static SolverSession make(Args&&... args)
   {
      return SolverSession{new Model(std::forward< Args >(args)...), &Model::ops};
   }
};

template < typename TypedNode >
struct ErasedPolicyNodeBackend final: PolicyNodeView::Backend {
   explicit ErasedPolicyNodeBackend(
      TypedNode concrete_node,
      ErasedInfoState concrete_key,
      Player concrete_owner,
      PolicyViewKind concrete_kind
   )
       : node(std::move(concrete_node)),
         key(std::move(concrete_key)),
         owner(concrete_owner),
         policy_kind(concrete_kind)
   {
   }

   TypedNode node;
   ErasedInfoState key;
   Player owner = Player::unknown;
   PolicyViewKind policy_kind = PolicyViewKind::current;

   [[nodiscard]] bool valid() const noexcept final { return node.valid(); }
   [[nodiscard]] size_t generation() const noexcept final { return node.generation(); }
   [[nodiscard]] PolicyViewKind kind() const noexcept final { return policy_kind; }
   [[nodiscard]] Player player() const final
   {
      if(not valid())
         throw std::logic_error("ErasedPolicyNodeBackend is stale");
      return owner;
   }

   [[nodiscard]] ErasedInfoState info_state() const final
   {
      if(not valid())
         throw std::logic_error("ErasedPolicyNodeBackend is stale");
      return key;
   }

   [[nodiscard]] size_t size() const final { return node.size(); }

   [[nodiscard]] ErasedAction action_at(size_t index) const final
   {
      return ErasedValue::action(node.action_at(index));
   }

   [[nodiscard]] double value_at(size_t index) const final { return node.value_at(index); }

   [[nodiscard]] std::optional< double > find(const ErasedAction& action) const final
   {
      if(not action.valid() or action.kind() != ValueKind::action
         or not action.template holds< typename TypedNode::action_type >()) {
         return std::nullopt;
      }
      return node.find(*action.template get_if< typename TypedNode::action_type >());
   }

   [[nodiscard]] bool contains(const ErasedAction& action) const final
   {
      if(not action.valid() or action.kind() != ValueKind::action
         or not action.template holds< typename TypedNode::action_type >()) {
         return false;
      }
      return node.contains(*action.template get_if< typename TypedNode::action_type >());
   }
};

template < typename TypedLookup, typename InfoState, typename Action >
struct ErasedPolicyLookupBackend final: PolicyLookup::Backend {
   explicit ErasedPolicyLookupBackend(TypedLookup concrete_lookup)
       : lookup(std::move(concrete_lookup))
   {
   }

   TypedLookup lookup;

   [[nodiscard]] bool valid() const noexcept final { return lookup.valid(); }
   [[nodiscard]] size_t generation() const noexcept final { return lookup.generation(); }

   template < rm::PolicyLabel Label >
   [[nodiscard]] std::optional< PolicyNodeView > find_label(const ErasedInfoState& key) const
   {
      if(not key.valid() or key.kind() != ValueKind::info_state
         or not key.template holds< InfoState >()) {
         return std::nullopt;
      }
      auto found = lookup.template find< Label >(*key.template get_if< InfoState >());
      if(not found) {
         return std::nullopt;
      }
      using typed_node_type = std::remove_cvref_t< decltype(*found) >;
      using backend_type = ErasedPolicyNodeBackend< typed_node_type >;
      return PolicyNodeView::from_backend(std::make_shared< backend_type >(
         std::move(*found),
         key,
         key.template get_if< InfoState >()->player(),
         Label == rm::PolicyLabel::current ? PolicyViewKind::current : PolicyViewKind::average
      ));
   }

   [[nodiscard]] std::optional< PolicyNodeView >
   find(PolicyViewKind kind, const ErasedInfoState& key) const final
   {
      switch(kind) {
         case PolicyViewKind::current: return find_label< rm::PolicyLabel::current >(key);
         case PolicyViewKind::average: return find_label< rm::PolicyLabel::average >(key);
      }
      return std::nullopt;
   }

   template < rm::PolicyLabel Label >
   size_t visit_label(const std::function< void(const PolicyNodeView&) >& visitor) const
   {
      return lookup.template visit< Label >([&](const InfoState& key, const auto& node) {
         using typed_node_type = std::remove_cvref_t< decltype(node) >;
         using backend_type = ErasedPolicyNodeBackend< typed_node_type >;
         auto erased_node = PolicyNodeView::from_backend(std::make_shared< backend_type >(
            node,
            ErasedValue::info_state(key),
            key.player(),
            Label == rm::PolicyLabel::current ? PolicyViewKind::current : PolicyViewKind::average
         ));
         visitor(erased_node);
      });
   }

   size_t visit(PolicyViewKind kind, const std::function< void(const PolicyNodeView&) >& visitor)
      const final
   {
      switch(kind) {
         case PolicyViewKind::current: return visit_label< rm::PolicyLabel::current >(visitor);
         case PolicyViewKind::average: return visit_label< rm::PolicyLabel::average >(visitor);
      }
      return 0;
   }
};

template < typename TypedLookup, typename InfoState, typename Action >
[[nodiscard]] PolicyView erase_policy_lookup(TypedLookup lookup, PolicyViewKind kind)
{
   using backend_type = ErasedPolicyLookupBackend<
      std::remove_cvref_t< TypedLookup >,
      InfoState,
      Action >;
   return PolicyView::from_backend(std::make_shared< backend_type >(std::move(lookup)), kind);
}

template < typename RootMap >
[[nodiscard]] IterationResult make_iteration_result(size_t iteration, const RootMap& root)
{
   IterationResult result{.iteration = iteration, .root_values = {}};
   for(const auto [player, value] : root.get()) {
      result.root_values.push_back(RootValue{.player = player, .value = value});
   }
   std::ranges::sort(result.root_values, [](const RootValue& left, const RootValue& right) {
      return static_cast< int >(left.player) < static_cast< int >(right.player);
   });
   return result;
}

template < typename RootValues >
[[nodiscard]] TraceResult make_trace_result(
   size_t first_iteration,
   size_t last_iteration,
   size_t every,
   const RootValues& roots
)
{
   TraceResult result{
      .first_iteration = first_iteration, .last_iteration = last_iteration, .iterations = {}};
   result.iterations.reserve(roots.size());
   for(size_t index = 0; index < roots.size(); ++index) {
      result.iterations.push_back(
         make_iteration_result(first_iteration + (index + 1) * every - 1, roots[index])
      );
   }
   return result;
}

template < typename Env >
[[nodiscard]] size_t
actual_player_count(const Env& environment, const typename Env::world_state_type& root_state)
{
   const auto players = environment.players(root_state);
   return static_cast< size_t >(std::ranges::count_if(players, [](Player player) {
      return player != Player::chance;
   }));
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

   /**
    * @brief Run one erased operation and translate every possible failure into a Result.
    *
    * This is the outermost frame of the compiled session: nothing above it can catch a C++
    * exception, and a foreign (non-std::exception) throw escaping here would cross the erased
    * function-pointer boundary into a caller that has no handler for it. A dynamic provider's
    * language runtime is exactly such a source, so the catch-all is not defensive padding.
    */
   template < typename T, typename Operation >
   [[nodiscard]] static Result< T > guarded(const char* what, Operation&& operation)
   {
      try {
         return std::invoke(std::forward< Operation >(operation));
      } catch(const DynamicProviderError& violation) {
         return operation_error< T >(
            CapabilityErrorCode::invalid_dynamic_provider,
            std::string("solver ") + what
               + " rejected a dynamic provider value: " + violation.what()
         );
      } catch(const std::exception& exception) {
         return operation_error< T >(
            CapabilityErrorCode::session_failure,
            std::string("solver ") + what + " failed: " + exception.what()
         );
      } catch(...) {
         return operation_error< T >(
            CapabilityErrorCode::session_failure,
            std::string("solver ") + what + " failed with a non-standard exception"
         );
      }
   }

   static Result< IterationResult > iterate(void* object)
   {
      auto& self = *static_cast< SessionModel* >(object);
      return guarded< IterationResult >("iteration", [&] {
         const size_t iteration = self.solver.iteration();
         return make_iteration_result(iteration, self.solver.iterate());
      });
   }

   static Result< void > advance(void* object, size_t iterations)
   {
      auto& self = *static_cast< SessionModel* >(object);
      return guarded< void >("advance", [&]() -> Result< void > {
         self.solver.advance(iterations);
         return {};
      });
   }

   static Result< std::optional< IterationResult > > advance_last(void* object, size_t iterations)
   {
      auto& self = *static_cast< SessionModel* >(object);
      return guarded< std::optional< IterationResult > >("advance_last", [&] {
         const size_t first_iteration = self.solver.iteration();
         auto root = self.solver.advance_last(iterations);
         if(not root) {
            return std::optional< IterationResult >{};
         }
         return std::optional< IterationResult >{
            make_iteration_result(first_iteration + iterations - 1, *root)};
      });
   }

   static Result< TraceResult > trace(void* object, size_t iterations, size_t every)
   {
      auto& self = *static_cast< SessionModel* >(object);
      if(every == 0) {
         return operation_error< TraceResult >(
            CapabilityErrorCode::invalid_spec, "trace cadence must be greater than zero"
         );
      }
      return guarded< TraceResult >("trace", [&] {
         const size_t first_iteration = self.solver.iteration();
         auto roots = self.solver.trace(iterations, every);
         return make_trace_result(first_iteration, self.solver.iteration(), every, roots);
      });
   }

   static Result< SessionStats > stats(const void* object)
   {
      const auto& self = *static_cast< const SessionModel* >(object);
      return guarded< SessionStats >("statistics", [&] {
         size_t current_entries = 0;
         size_t average_entries = 0;
         self.solver.visit_current_policy([&](const auto&, const auto& node) {
            current_entries += node.size();
         });
         self.solver.visit_average_policy([&](const auto&, const auto& node) {
            average_entries += node.size();
         });
         return SessionStats{
            .game = Game,
            .solver = SolverFamily,
            .profile = Profile,
            .iteration = self.solver.iteration(),
            .cycle = self.solver.cycle(),
            .player_count = actual_player_count(self.solver.env(), self.solver.root_state()),
            .current_policy_entries = current_entries,
            .average_policy_entries = average_entries};
      });
   }

   static Result< PolicyView > policy_lookup(const void* object, PolicyViewKind kind)
   {
      const auto& self = *static_cast< const SessionModel* >(object);
      return guarded< PolicyView >("policy lookup", [&] {
         return erase_policy_lookup<
            decltype(self.solver.policy_lookup()),
            typename Solver::info_state_type,
            typename Solver::action_type >(self.solver.policy_lookup(), kind);
      });
   }

   inline static constexpr SolverSessionOps ops{
      .destroy = &destroy,
      .iterate = &iterate,
      .advance = &advance,
      .advance_last = &advance_last,
      .trace = &trace,
      .stats = &stats,
      .policy_lookup = &policy_lookup};
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

/**
 * @brief The one static session thunk.
 *
 * A CapabilityDescriptor exposes this as a plain function pointer, so a caller that already holds
 * a descriptor reaches this code without passing through make_session(). Every precondition that
 * make_session() checks is therefore re-checked here rather than assumed.
 */
template < typename Game, typename Profile >
[[nodiscard]] Result< SolverSession >
make_session_impl(const GameHandle& handle, SessionOptions options)
{
   // A descriptor for an inadmissible pair is never emitted, so this branch is unreachable through
   // the registry. Stating it as a constant-evaluated condition is what keeps the concrete solver
   // instantiation out of this translation unit for a pair the compiler did not admit.
   static_assert(
      profile_supported< typename Game::env_type, Profile >(),
      "a static session thunk may only be instantiated for a compiler-admitted game/profile pair"
   );

   if(handle.game_id() != Game::id) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_handle,
         .message = "GameHandle ID does not match its selected capability thunk",
         .game = handle.game_id(),
         .solver = Profile::solver,
         .profile = Profile::id});
   }
   if(not std::isfinite(options.epsilon) or options.epsilon < 0. or options.epsilon > 1.) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::invalid_spec,
         .message = "sampling epsilon must be finite and in [0, 1]",
         .game = Game::id,
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
      auto root_state = initial_world_state< Game >(environment, *normalized);
      auto root = std::make_unique< auto_world_state_type< typename Game::env_type > >(
         std::move(root_state)
      );
      auto solver = make_concrete_solver< Profile >(
         std::move(environment), std::move(root), options
      );
      using solver_type = decltype(solver);
      using model_type = SessionModel< solver_type, Game::id, Profile::solver, Profile::id >;
      return SessionFactory::make< model_type >(std::move(solver));
   } catch(const std::exception& exception) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to create ") + std::string(profile_name< Profile >())
                    + " on " + std::string(game_name< Game >()) + ": " + exception.what(),
         .game = Game::id,
         .solver = Profile::solver,
         .profile = Profile::id});
   } catch(...) {
      return std::unexpected(CapabilityError{
         .code = CapabilityErrorCode::construction_failure,
         .message = std::string("failed to create ") + std::string(profile_name< Profile >())
                    + " on " + std::string(game_name< Game >())
                    + ": the solver threw a non-standard exception",
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
      .name = game_name< Game >(),
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
   return ProfileDescriptor{
      .id = Profile::id, .solver = Profile::solver, .name = profile_name< Profile >()};
}

template < typename... Profiles >
[[nodiscard]] consteval auto make_profile_descriptors(type_list< Profiles... >)
{
   static_assert(unique_values(std::array{Profiles::id...}), "static profile IDs must be unique");
   return std::array{make_profile_descriptor< Profiles >()...};
}

inline constexpr std::array solver_descriptors{
   SolverDescriptor{.id = SolverId::vanilla_cfr, .name = meta::enum_name(SolverId::vanilla_cfr)},
   SolverDescriptor{.id = SolverId::cfr_plus, .name = meta::enum_name(SolverId::cfr_plus)},
   SolverDescriptor{.id = SolverId::lazy_cfr, .name = meta::enum_name(SolverId::lazy_cfr)},
   SolverDescriptor{
      .id = SolverId::lazy_cfr_plus,
      .name = meta::enum_name(SolverId::lazy_cfr_plus)},
   SolverDescriptor{
      .id = SolverId::extragradient_cfr,
      .name = meta::enum_name(SolverId::extragradient_cfr)},
   SolverDescriptor{
      .id = SolverId::discounted_cfr,
      .name = meta::enum_name(SolverId::discounted_cfr)},
   SolverDescriptor{.id = SolverId::linear_cfr, .name = meta::enum_name(SolverId::linear_cfr)},
   SolverDescriptor{
      .id = SolverId::exponential_cfr,
      .name = meta::enum_name(SolverId::exponential_cfr)},
   SolverDescriptor{.id = SolverId::greedy_cfr, .name = meta::enum_name(SolverId::greedy_cfr)},
   SolverDescriptor{.id = SolverId::mccfr, .name = meta::enum_name(SolverId::mccfr)},
   SolverDescriptor{.id = SolverId::mccfr_plus, .name = meta::enum_name(SolverId::mccfr_plus)}};

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

template < typename Game, typename... Profiles >
[[nodiscard]] consteval auto make_game_capabilities(type_list< Profiles... > profiles)
{
   (void) profiles;
   constexpr size_t count = capability_count_for< Game >(type_list< Profiles... >{});
   std::array< CapabilityDescriptor, count > output{};
   size_t index = 0;
   append_game_capabilities< Game >(output, index, profiles);
   return output;
}

/**
 * @brief The one-game unit emitted by a compiled binding-runtime partition.
 *
 * The function-local static arrays live in the partition translation unit that instantiates this
 * template, and each partition instantiates concrete solvers for one game only. What partitioning
 * removes is the *instantiation* cross-product, not the admissibility work: every partition still
 * evaluates profile_supported() for every profile, because that is what decides which of its
 * capabilities exist. Those checks are cheap constant evaluation; the expensive part that stays
 * partitioned is the concrete solver each admitted pair instantiates.
 *
 * Public consumers see only spans returned by catalog(), so including types.hpp never emits a
 * game/solver pair.
 */
struct CatalogPartition {
   GameDescriptor game{};
   std::span< const CapabilityDescriptor > capabilities{};
};

template < typename Game >
[[nodiscard]] const CatalogPartition& partition_for() noexcept
{
   static_assert(
      list_contains< Game >(game_types{}),
      "a partition may only be emitted for a game tag that is part of the static game list"
   );
   static constexpr auto game = make_game_descriptor< Game >();
   static constexpr auto game_capabilities = make_game_capabilities< Game >(profile_types{});
   static_assert(
      unique_field_ids(Game::fields), "fields within a static game specification must be unique"
   );
   static_assert(
      unique_capabilities(game_capabilities),
      "a compiled game partition must contain unique admitted capabilities"
   );
   static const CatalogPartition value{
      .game = game, .capabilities = std::span< const CapabilityDescriptor >{game_capabilities}};
   return value;
}

}  // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// public lookup API ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace nor::binding::runtime

#endif  // NOR_BINDING_RUNTIME_CATALOG_HPP
