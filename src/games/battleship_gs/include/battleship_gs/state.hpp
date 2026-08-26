
#ifndef NOR_BATTLESHIP_GS_STATE_HPP
#define NOR_BATTLESHIP_GS_STATE_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <variant>
#include <vector>

#include "common/common.hpp"

namespace battleship_gs {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent
enum class Player : int8_t { none = -2, one = 0, two = 1 };

[[nodiscard]] constexpr Player opponent(Player player)
{
   return player == Player::one ? Player::two : Player::one;
}

template < typename To = size_t, typename T >
[[nodiscard]] constexpr To as_int(T value)
{
   return static_cast< To >(value);
}

// #####################################################################################################################
// rule transcription notes (canonical literature source)
// #####################################################################################################################
//
// SOURCE. Farina, Ling, Fang, Sandholm, "Correlation in Extensive-Form Games: Saddle-Point
// Formulation and Benchmarks", NeurIPS 2019 (arXiv:1905.12564), Section 5.1 and Appendix E.1 --
// the paper that introduced this game as one of the two canonical GENERAL-SUM imperfect-
// information benchmarks for the extensive-form-correlated-equilibrium literature. NOTE that this
// is a DIFFERENT game from the zero-sum Battleship of Farina et al., AAAI 2021 (PCFR+, App. G),
// which lives in src/games/battleship: the EFCE benchmark parameterizes heterogeneous ordered
// fleets and a loss multiplier gamma >= 1 that makes the game genuinely general-sum.
//
// PARAMETERIZATION (Appendix E.1). A game is a tuple (H, W, S, r, gamma):
// - H, W >= 1: height/width of EACH player's private playing field;
// - S = [(l_0, v_0), ..., (l_{k-1}, v_{k-1})]: the ORDERED fleet shared by both players, ship i
//   having length l_i and value v_i;
// - r >= 1: the shot budget of every player ("rounds");
// - gamma >= 1: the LOSS MULTIPLIER controlling the relative value of losing vs destroying ships.
//
// PROGRESSION (Appendix E.1). Placement phase: players take turns (Player 1 starts) secretly
// placing their ships on their own field, in the order in which they appear in S (ship-by-ship,
// cf. the sibling zero-sum engine); ships lie horizontally or vertically within the field and may
// not overlap other ships of the SAME player; locations are private information. Shooting phase:
// players alternate firing single shots at coordinates of the opponent's field (Player 1 first);
// after every shot the referee publicly announces hit / miss, plus the destruction of a ship once
// ALL of its cells have been hit -- the IDENTITY of the destroyed ship is not revealed. The game
// ends when one player has lost ALL their ships or every player has fired r shots, whichever
// comes first.
//
// NO-REPEAT SHOOTING (repository rule, transcribed as dominance pruning). Appendix E.1 places no
// explicit restriction on repeated shots, but re-targeting an already-fired cell wastes a shot
// under perfect recall and is therefore strictly dominated at EVERY infoset of the benchmark
// family: it can never be part of any Nash equilibrium's support, and uniform play over the
// repeat-pruned action sets reproduces the published reference-instance statistics EXACTLY
// (P(player one sinks) = 5/9, P(player two sinks) = 1/3, P(peaceful) = 1/9, SW = -8/9; Section
// 5.1) -- with repeats allowed they do not (one obtains 13/27, 26/81, 16/81). This engine hence
// never offers a previously fired cell while fresh targets remain (fresh targets cannot run out
// before terminality: exhausting them would mean having covered the whole opposing field,
// which already sank the opponent's fleet).
//
// PAYOFFS (Appendix E.1): "for each opponent's ship that the player has destroyed, the player
// receives a payoff equal to the value v of that ship; for each ship that the player has lost to
// the opponent, the player incurs a negative payoff equal to gamma * v". I.e.
//     u_i = sum_{destroyed opponents' ships} v_j  -  gamma * sum_{own destroyed ships} v_i.
// For gamma = 1 this recovers the zero-sum variant; any gamma > 1 makes the game GENERAL SUM.
//
// REFERENCE INSTANCE (Section 5.1). Board 3 x 1, one ship of value and length 1 per player,
// r = 2 shots per player, gamma = 2. Under the SW-maximizing Nash equilibrium (uniform placement
// + uniformly random shooting among fresh targets) the sink probabilities are 5/9 (player one) /
// 1/3 (player two) with a peaceful outcome probability of 1/9 and social welfare -8/9; these
// published numbers are asserted in the unit tests.

/// the maximal number of cells of a playing field's flattened grid (limits rows * cols)
constexpr size_t max_cells = 64;
/// the maximal length of a single ship
constexpr size_t max_ship_length = 6;

/**
 * @brief A single grid cell with row-major coordinates.
 */
struct Cell {
   int8_t row = 0;
   int8_t col = 0;

   friend bool operator==(const Cell&, const Cell&) = default;
   friend bool operator!=(const Cell&, const Cell&) = default;
};

/**
 * @brief One entry of the ordered fleet specification: a ship of `length` cells worth `value`.
 */
struct ShipSpec {
   uint8_t length = 1;
   double value = 1.;

   friend bool operator==(const ShipSpec&, const ShipSpec&) = default;
   friend bool operator!=(const ShipSpec&, const ShipSpec&) = default;
};

/**
 * @brief Placement of the CURRENT ship (the next unfilled fleet slot) as a straight line of
 * cells starting at 'start' and stepping unit-wise along (drow, dcol).
 *
 * The placed length is implied by the fleet order (both players place the very same ship spec at
 * any point of the alternating sequence), so the action itself carries no length payload. Only
 * the canonical orientations (+row) / (+col) are generated and accepted.
 */
struct Place {
   Cell start{};
   int8_t drow = 0;
   int8_t dcol = 1;

   friend bool operator==(const Place&, const Place&) = default;
   friend bool operator!=(const Place&, const Place&) = default;

   /// the i-th covered cell
   [[nodiscard]] Cell cell(size_t index) const
   {
      return Cell{int8_t(start.row + index * drow), int8_t(start.col + index * dcol)};
   }
};

/**
 * @brief Firing a shot at cell 'target'.
 *
 * While fresh targets remain, cells this player already fired at are neither enumerated by
 * actions() nor accepted by is_valid() -- wasting a shot on a known miss is strictly dominated
 * and repeat-free play is what reproduces the published Section 5.1 statistics exactly.
 */
struct Fire {
   Cell target{};

   friend bool operator==(const Fire&, const Fire&) = default;
   friend bool operator!=(const Fire&, const Fire&) = default;
};

using Action = std::variant< Place, Fire >;

/// the phase of the game: secret alternating placements followed by alternating fire
enum class Phase : uint8_t { one_placement = 0, two_placement, one_fire, two_fire, over };

/**
 * @brief Configuration of a general-sum Battleship instance (Farina et al. 2019, App. E.1).
 *
 * Defaults transcribe the paper's reference instance of Section 5.1: a 3x1 board, one length-1
 * ship of value 1 per player, r = 2 shots per player, loss multiplier gamma = 2.
 */
struct Config {
   size_t rows = 3;  //< height H of every player's field
   size_t cols = 1;  //< width W of every player's field
   /// the ordered fleet spec shared by both players
   std::vector< ShipSpec > fleet{{1u, 1.}};
   size_t max_shots = 2;  //< the shot budget r of every player
   double loss_multiplier = 2.;  //< the loss multiplier gamma >= 1

   Config() = default;
   Config(
      size_t rows_,
      size_t cols_,
      std::vector< ShipSpec > fleet_,
      size_t max_shots_,
      double loss_multiplier_
   )
       : rows(rows_),
         cols(cols_),
         fleet(std::move(fleet_)),
         max_shots(max_shots_),
         loss_multiplier(loss_multiplier_)
   {
      validate();
   }

   [[nodiscard]] size_t cell_count() const { return rows * cols; }
   [[nodiscard]] size_t fleet_size() const { return fleet.size(); }
   [[nodiscard]] size_t fleet_cells() const
   {
      return std::accumulate(
         fleet.begin(),
         fleet.end(),
         size_t{0},
         [](size_t acc, const ShipSpec& s) { return acc + s.length; }
      );
   }
   [[nodiscard]] size_t ship_length(size_t ship_index) const { return fleet.at(ship_index).length; }
   [[nodiscard]] double ship_value(size_t ship_index) const { return fleet.at(ship_index).value; }

   void validate() const
   {
      if(rows == 0 or cols == 0) {
         throw std::invalid_argument("battleship_gs grid dimensions must be positive.");
      }
      if(cell_count() > max_cells) {
         throw std::invalid_argument(
            "battleship_gs fields support at most " + std::to_string(max_cells) + " cells."
         );
      }
      if(fleet.empty()) {
         throw std::invalid_argument("battleship_gs fleets must hold at least one ship.");
      }
      if(fleet.size() > cell_count()) {
         throw std::invalid_argument(
            "battleship_gs fleets cannot hold more ships than there are cells."
         );
      }
      for(const auto& [length, value] : fleet) {
         if(length == 0 or length > max_ship_length) {
            throw std::invalid_argument(
               "battleship_gs supports ships of length 1 to " + std::to_string(max_ship_length)
               + "."
            );
         }
         if(length > std::max(rows, cols)) {
            throw std::invalid_argument("battleship_gs ships must fit inside the field.");
         }
         if(not (value > 0.)) {
            throw std::invalid_argument("battleship_gs ship values must be positive.");
         }
      }
      if(fleet_cells() > cell_count()) {
         throw std::invalid_argument("battleship_gs fleets hold more cells than the field provides."
         );
      }
      if(max_shots == 0) {
         throw std::invalid_argument("battleship_gs requires a positive shot budget.");
      }
      if(loss_multiplier < 1.) {
         throw std::invalid_argument(
            "battleship_gs requires a loss multiplier gamma >= 1 (Appendix E.1); smaller values "
            "are outside the benchmark family."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/// the canonical reference instance of Farina et al. 2019, Section 5.1:
/// Battleship(H=3, W=1, S={(1, 1)}, r=2, gamma=2)
[[nodiscard]] inline Config canonical_config()
{
   return Config{};
}

/**
 * @brief The world state of a single deterministic general-sum battleship game.
 *
 * Hidden information arises purely from the players' secretly placed fleets: during the
 * placement phases the acting player commits his ship while the public observation carries no
 * positional payload. Afterwards both players alternate firing shots which are resolved publicly
 * by the referee (hit/miss/sink without ship identity).
 *
 * All data is kept in small fixed-size/fixed-length containers so copying (which happens for
 * every edge of the search tree) stays cheap.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(std::move(config)) { m_config.validate(); }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   void apply_action(const Action& action);

   [[nodiscard]] bool is_valid(const Action& action) const;

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_phase == Phase::over; }

   /**
    * payoff of `player` = sum of values of the opponent's sunk ships minus the loss-multiplier
    * weighted sum of values of own sunk ships (zero for a timeout without any sink).
    */
   [[nodiscard]] double payoff(Player player) const;

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   [[nodiscard]] std::vector< Action > actions(Player player) const;

   [[nodiscard]] Phase phase() const { return m_phase; }
   [[nodiscard]] bool placing() const
   {
      return m_phase == Phase::one_placement || m_phase == Phase::two_placement;
   }
   [[nodiscard]] bool firing() const
   {
      return m_phase == Phase::one_fire || m_phase == Phase::two_fire;
   }
   [[nodiscard]] Player active_player() const
   {
      switch(m_phase) {
         case Phase::one_placement:
         case Phase::one_fire: return Player::one;
         case Phase::two_placement:
         case Phase::two_fire: return Player::two;
         default: return Player::none;
      }
   }
   [[nodiscard]] const Config& config() const { return m_config; }

   /// the fleet of `player`, ship-major: m_fleets[i][j] = j-th cell of his j-th placed ship
   [[nodiscard]] const auto& fleet_cells(Player player) const { return m_fleets[as_int(player)]; }
   [[nodiscard]] size_t ships_placed(Player player) const { return m_placed_ships[as_int(player)]; }
   /// the length of the ship that is due to be placed next (only meaningful while placing)
   [[nodiscard]] size_t pending_ship_length(Player placer) const
   {
      return m_config.ship_length(m_placed_ships[as_int(placer)]);
   }
   /// whether every ship of `player`'s (fully placed) fleet has been fully hit by
   /// `opponent(player)`
   [[nodiscard]] bool fleet_sunk(Player player) const;
   /// whether the i-th ship of `player`'s fleet has been fully hit by the opponent
   [[nodiscard]] bool ship_sunk(Player player, size_t ship_index) const;
   [[nodiscard]] size_t shots_used(Player player) const { return m_shots[as_int(player)].size(); }
   /// the cells `player` fired at, in firing order (chronological)
   [[nodiscard]] const auto& shot_log(Player player) const { return m_shots[as_int(player)]; }
   /// whether `shooter`'s shots have scored a hit at `target` so far
   [[nodiscard]] bool was_hit(Player shooter, Cell target) const
   {
      return m_hits[as_int(shooter)][cell_index(target)];
   }
   [[nodiscard]] bool was_fired_at(Player shooter, Cell target) const
   {
      const auto& log = m_shots[as_int(shooter)];
      return std::find(log.begin(), log.end(), target) != log.end();
   }

   [[nodiscard]] bool occupies(Player owner, Cell cell) const
   {
      for(const auto& ship : m_fleets[as_int(owner)]) {
         if(std::ranges::find(ship, cell) != ship.end()) {
            return true;
         }
      }
      return false;
   }

   [[nodiscard]] size_t cell_index(Cell cell) const
   {
      return size_t(cell.row) * m_config.cols + size_t(cell.col);
   }
   [[nodiscard]] Cell make_cell(size_t index) const
   {
      return Cell{int8_t(index / m_config.cols), int8_t(index % m_config.cols)};
   }
   [[nodiscard]] bool in_grid(Cell cell) const
   {
      return cell.row >= 0 and cell.col >= 0 and size_t(cell.row) < m_config.rows
             and size_t(cell.col) < m_config.cols;
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_phase == other.m_phase && m_fleets == other.m_fleets
             && m_placed_ships == other.m_placed_ships && m_hits == other.m_hits
             && m_shots == other.m_shots;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   [[nodiscard]] bool _overlaps_own_fleet(Player placer, const Place& place) const;
   /// whether every ship of `fleet_owner`'s (fully placed) fleet is fully hit
   [[nodiscard]] bool _all_ships_sunk_of(Player fleet_owner) const;
   /// whether `shooter` has already fired at every cell of the opposing field (defensive
   /// completeness only: inside a live game this cannot happen before his own fleet loss ends
   /// the game, since covering all cells necessarily sank every opposing ship)
   [[nodiscard]] bool _fresh_targets_exhausted(Player shooter) const
   {
      return m_shots[as_int(shooter)].size() >= m_config.cell_count();
   }

   Config m_config;
   Phase m_phase = Phase::one_placement;
   std::array< std::vector< std::vector< Cell > >, 2 > m_fleets{};
   std::array< size_t, 2 > m_placed_ships{0, 0};
   /// cells (bit per grid index) that this player's shots have hit inside the opponent's fleet
   std::array< std::bitset< max_cells >, 2 > m_hits{};
   /// the cells each player fired at, in chronological firing order
   std::array< std::vector< Cell >, 2 > m_shots{};
};

}  // namespace battleship_gs

namespace battleship_gs {

inline void State::apply_action(const Action& action)
{
   if(m_phase == Phase::over) {
      throw std::logic_error("battleship_gs state is terminal; no further actions can be applied.");
   }
   if(placing()) {
      if(not std::holds_alternative< Place >(action)) {
         throw std::invalid_argument("battleship_gs placement phases only accept 'Place' actions.");
      }
      if(not is_valid(action)) {
         throw std::invalid_argument("illegal battleship_gs ship placement.");
      }
      auto placer = active_player();
      const auto& place = std::get< Place >(action);
      const auto length = pending_ship_length(placer);
      auto& fleet = m_fleets[as_int(placer)];
      fleet.emplace_back();
      fleet.back().reserve(length);
      for(size_t i : std::views::iota(size_t{0}, length)) {
         fleet.back().push_back(place.cell(i));
      }
      m_placed_ships[as_int(placer)] += 1;
      if(m_placed_ships[0] + m_placed_ships[1] == 2 * m_config.fleet_size()) {
         // both fleets are fully placed --> the shooting duel begins with player one
         m_phase = Phase::one_fire;
      } else {
         // placements alternate secretly ship-by-ship
         m_phase = placer == Player::one ? Phase::two_placement : Phase::one_placement;
      }
      return;
   }

   // firing phases
   if(not std::holds_alternative< Fire >(action)) {
      throw std::invalid_argument("battleship_gs firing phases only accept 'Fire' actions.");
   }
   if(not is_valid(action)) {
      throw std::invalid_argument("illegal battleship_gs shot.");
   }
   const auto shooter = active_player();
   const auto target = std::get< Fire >(action).target;
   m_shots[as_int(shooter)].push_back(target);
   if(occupies(opponent(shooter), target)) {
      m_hits[as_int(shooter)][cell_index(target)] = true;
   }
   if(_all_ships_sunk_of(opponent(shooter))) {
      // a fully destroyed opponent fleet ends the game immediately
      m_phase = Phase::over;
   } else if(m_shots[0].size() >= m_config.max_shots and m_shots[1].size() >= m_config.max_shots) {
      // both shot budgets are exhausted without a decision
      m_phase = Phase::over;
   } else {
      m_phase = shooter == Player::one ? Phase::two_fire : Phase::one_fire;
   }
}

inline bool State::is_valid(const Action& action) const
{
   if(placing()) {
      const auto* place = std::get_if< Place >(&action);
      if(place == nullptr) {
         return false;
      }
      // canonical orientations only (right neighbour / down neighbour stepping)
      if(not ((place->drow == 0 and place->dcol == 1) or (place->drow == 1 and place->dcol == 0))) {
         return false;
      }
      const auto length = pending_ship_length(active_player());
      const auto last = place->cell(length - 1);
      if(not in_grid(place->start) or not in_grid(last)) {
         return false;
      }
      return not _overlaps_own_fleet(active_player(), *place);
   }
   if(firing()) {
      const auto* fire = std::get_if< Fire >(&action);
      if(fire == nullptr or not in_grid(fire->target)) {
         return false;
      }
      // a shooter never re-targets one of his own already-fired cells while fresh ones remain
      return not was_fired_at(active_player(), fire->target)
             or _fresh_targets_exhausted(active_player());
   }
   return false;
}

inline double State::payoff(Player player) const
{
   const auto foe = opponent(player);
   double own_lost = 0.;
   double foe_sunk = 0.;
   for(size_t ship : std::views::iota(size_t{0}, m_placed_ships[as_int(player)])) {
      if(ship_sunk(player, ship)) {
         own_lost += m_config.ship_value(ship);
      }
   }
   for(size_t ship : std::views::iota(size_t{0}, m_placed_ships[as_int(foe)])) {
      if(ship_sunk(foe, ship)) {
         foe_sunk += m_config.ship_value(ship);
      }
   }
   return foe_sunk - m_config.loss_multiplier * own_lost;
}

inline std::vector< Action > State::actions(Player player) const
{
   std::vector< Action > out;
   if(m_phase == Phase::over or active_player() != player) {
      return out;
   }
   if(placing()) {
      const auto length = pending_ship_length(player);
      out.reserve(2 * m_config.cell_count());
      for(size_t idx : std::views::iota(size_t{0}, m_config.cell_count())) {
         auto start = make_cell(idx);
         // canonical orientations (rightwards / downwards lines) to avoid duplicates
         if(length == 1) {
            // a single cell has no orientation; fix the horizontal encoding as its unique
            // canonical form so every placement is generated exactly once
            out.emplace_back(Place{start, int8_t{0}, int8_t{1}});
            continue;
         }
         if(size_t(start.col) + length <= m_config.cols) {
            out.emplace_back(Place{start, int8_t{0}, int8_t{1}});
         }
         if(size_t(start.row) + length <= m_config.rows) {
            out.emplace_back(Place{start, int8_t{1}, int8_t{0}});
         }
      }
      std::erase_if(out, [&](const Action& action) { return not is_valid(action); });
      return out;
   }
   out.reserve(m_config.cell_count());
   const bool fresh_exhausted = _fresh_targets_exhausted(player);
   for(size_t idx : std::views::iota(size_t{0}, m_config.cell_count())) {
      const auto cell = make_cell(idx);
      if(not fresh_exhausted and was_fired_at(player, cell)) {
         continue;
      }
      out.emplace_back(Fire{cell});
   }
   return out;
}

inline bool State::fleet_sunk(Player player) const
{
   return _all_ships_sunk_of(player);
}

inline bool State::ship_sunk(Player player, size_t ship_index) const
{
   const auto& ship = m_fleets[as_int(player)].at(ship_index);
   const auto& foe_hits = m_hits[as_int(opponent(player))];
   return std::ranges::all_of(ship, [&](Cell cell) { return foe_hits[cell_index(cell)]; });
}

inline bool State::_overlaps_own_fleet(Player placer, const Place& place) const
{
   const auto& fleet = m_fleets[as_int(placer)];
   const auto length = pending_ship_length(placer);
   for(size_t i : std::views::iota(size_t{0}, length)) {
      auto cell = place.cell(i);
      for(const auto& ship : fleet) {
         if(std::ranges::find(ship, cell) != ship.end()) {
            return true;
         }
      }
   }
   return false;
}

inline bool State::_all_ships_sunk_of(Player fleet_owner) const
{
   if(m_placed_ships[as_int(fleet_owner)] < m_config.fleet_size()) {
      // the fleet is not fully placed yet
      return false;
   }
   for(size_t ship : std::views::iota(size_t{0}, m_placed_ships[as_int(fleet_owner)])) {
      if(not ship_sunk(fleet_owner, ship)) {
         return false;
      }
   }
   return true;
}

}  // namespace battleship_gs

#endif  // NOR_BATTLESHIP_GS_STATE_HPP
