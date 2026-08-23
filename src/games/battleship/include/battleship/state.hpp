
#ifndef NOR_BATTLESHIP_STATE_HPP
#define NOR_BATTLESHIP_STATE_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include "common/common.hpp"

namespace battleship {

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

/// the maximal number of cells of the board's flattened grid (limits rows * cols)
constexpr size_t max_cells = 64;

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
 * @brief Placement of a size-2 ship occupying the two adjacent cells 'a' and 'b'.
 */
struct Place {
   Cell a{};
   Cell b{};

   friend bool operator==(const Place&, const Place&) = default;
   friend bool operator!=(const Place&, const Place&) = default;
};

/**
 * @brief Firing a shot at cell 'target'.
 *
 * Shots at already-visited cells are permitted (and simply re-resolve) so that fixed-horizon
 * instances with R > number-of-cells (e.g. Battleship(3) on the 3x2 grid) never dead-end.
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
 * @brief Configuration of a battleship instance.
 *
 * Two canonical families:
 * - "light": 2x2 or 2x3 grid, one ship per fleet, ship value 2, up to 3 shots per player.
 * - "classic-lite" (PCFR+ App. G): 3x2 grid, ships of size 2 and value 4 each, R shots per
 *   player with R in {3, 4}; also supports larger grids such as 4x3.
 */
struct Config {
   size_t rows = 2;  //< number of grid rows
   size_t cols = 2;  //< number of grid columns
   /// ships per fleet. NOTE: neither the PCFR+ (Farina et al., AAAI 2021, App. G) nor the
   /// Farina et al. NeurIPS'19 writeups state the fleet size explicitly; 2 is the de-facto
   /// standard used by published Battleship(3)/Battleship(4) experiments and hence the default
   size_t ships_per_fleet = 1;
   size_t max_shots = 3;  //< the shot budget R of every player
   double ship_value = 2.;  //< payoff value of every ship

   Config() = default;
   Config(
      size_t rows_,
      size_t cols_,
      size_t ships_per_fleet_,
      size_t max_shots_,
      double ship_value_
   )
       : rows(rows_),
         cols(cols_),
         ships_per_fleet(ships_per_fleet_),
         max_shots(max_shots_),
         ship_value(ship_value_)
   {
      validate();
   }

   [[nodiscard]] size_t cell_count() const { return rows * cols; }

   void validate() const
   {
      if(rows == 0 or cols == 0) {
         throw std::invalid_argument("battleship grid dimensions must be positive.");
      }
      if(cell_count() > max_cells) {
         throw std::invalid_argument(
            "battleship grids support at most " + std::to_string(max_cells) + " cells."
         );
      }
      if(ships_per_fleet == 0) {
         throw std::invalid_argument("battleship fleets must hold at least one ship.");
      }
      if(max_shots == 0) {
         throw std::invalid_argument("battleship requires a positive shot budget.");
      }
      if(ship_size() != 2) {
         throw std::invalid_argument(
            "this engine only supports ships of size 2 (two adjacent cells)."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;

  private:
   [[nodiscard]] size_t ship_size() const { return 2; }
};

/// the "light" variant family (Hennes et al./PDCFR-style BS instances):
/// one 1x2 ship worth 2 per fleet on a 2-row grid with <= 3 shots per player
[[nodiscard]] inline Config light_config(size_t cols = 2, size_t max_shots = 3)
{
   return Config{/*rows=*/2, cols, /*ships_per_fleet=*/1, max_shots, /*ship_value=*/2.};
}

/// the "classic-lite" variant family (PCFR+ App. G): ships of value 4, R-shot budgets,
/// e.g. Battleship(3)=(3,2,R=3), Battleship(4)=(3,2,R=4); the paper leaves the fleet size
/// unstated so it stays an explicit parameter (default 2)
[[nodiscard]] inline Config
classic_config(size_t rows, size_t cols, size_t max_shots, size_t ships_per_fleet = 2)
{
   return Config{rows, cols, ships_per_fleet, max_shots, /*ship_value=*/4.};
}

/**
 * @brief The world state of a single deterministic battleship game.
 *
 * Hidden information arises purely from the players' secretly placed fleets: during the
 * placement phases the acting player commits his ship while the public observation carries no
 * payload about its position. Afterwards both players alternate firing their shots which are
 * resolved publicly by the referee (hit/miss/sink).
 *
 * All data is kept in small fixed-size/fixed-length containers so copying (which happens for
 * every edge of the search tree) stays cheap.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(config) { m_config.validate(); }

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
    * payoff of `player` = sum of values of the opponent's sunk ships minus the sum of values
    * of own lost ships (zero for a timeout without any sink).
    */
   [[nodiscard]] double payoff(Player player) const;

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

   /// the fleet of `player` as a flat list of already-placed cells in placement order
   /// (size() == 2 * ships_placed(player))
   [[nodiscard]] const auto& fleet_cells(Player player) const { return m_fleets[as_int(player)]; }
   [[nodiscard]] size_t ships_placed(Player player) const { return m_placed_ships[as_int(player)]; }
   /// whether all ships of `player`'s fleet have been fully hit by `opponent(player)`
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
      const auto& fleet = m_fleets[as_int(owner)];
      for(size_t i = 0; i + 1 < fleet.size(); i += 2) {
         if(fleet[i] == cell or fleet[i + 1] == cell) {
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
   [[nodiscard]] bool _adjacent(Cell a, Cell b) const
   {
      return std::abs(int(a.row) - int(b.row)) + std::abs(int(a.col) - int(b.col)) == 1;
   }
   [[nodiscard]] bool _overlaps_own_fleet(Player placer, Cell a, Cell b) const;
   /// whether every ship of `fleet_owner`'s (fully placed) fleet is fully hit
   [[nodiscard]] bool _all_ships_sunk_of(Player fleet_owner) const;

   Config m_config;
   Phase m_phase = Phase::one_placement;
   std::array< std::vector< Cell >, 2 > m_fleets{};
   std::array< size_t, 2 > m_placed_ships{0, 0};
   /// cells (bit per grid index) that this player's shots have hit inside the opponent's fleet
   std::array< std::bitset< max_cells >, 2 > m_hits{};
   /// the cells each player fired at, in chronological firing order
   std::array< std::vector< Cell >, 2 > m_shots{};
};

}  // namespace battleship

namespace battleship {

inline void State::apply_action(const Action& action)
{
   if(m_phase == Phase::over) {
      throw std::logic_error("battleship state is terminal; no further actions can be applied.");
   }
   if(placing()) {
      if(not std::holds_alternative< Place >(action)) {
         throw std::invalid_argument("battleship placement phases only accept 'Place' actions.");
      }
      const auto& place = std::get< Place >(action);
      if(not is_valid(action)) {
         throw std::invalid_argument("illegal battleship ship placement.");
      }
      auto placer = active_player();
      auto& fleet = m_fleets[as_int(placer)];
      fleet.push_back(place.a);
      fleet.push_back(place.b);
      m_placed_ships[as_int(placer)] += 1;
      if(m_placed_ships[0] + m_placed_ships[1] == 2 * m_config.ships_per_fleet) {
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
      throw std::invalid_argument("battleship firing phases only accept 'Fire' actions.");
   }
   if(not is_valid(action)) {
      throw std::invalid_argument("illegal battleship shot.");
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
      return in_grid(place->a) && in_grid(place->b) && _adjacent(place->a, place->b)
             && not _overlaps_own_fleet(active_player(), place->a, place->b);
   }
   if(firing()) {
      const auto* fire = std::get_if< Fire >(&action);
      return fire != nullptr && in_grid(fire->target);
   }
   return false;
}

inline double State::payoff(Player player) const
{
   const auto foe = opponent(player);
   double own_lost = 0.;
   double foe_sunk = 0.;
   for(size_t ship = 0; ship < m_placed_ships[as_int(player)]; ++ship) {
      if(ship_sunk(player, ship)) {
         own_lost += m_config.ship_value;
      }
   }
   for(size_t ship = 0; ship < m_placed_ships[as_int(foe)]; ++ship) {
      if(ship_sunk(foe, ship)) {
         foe_sunk += m_config.ship_value;
      }
   }
   return foe_sunk - own_lost;
}

inline std::vector< Action > State::actions(Player player) const
{
   std::vector< Action > out;
   if(m_phase == Phase::over or active_player() != player) {
      return out;
   }
   if(placing()) {
      out.reserve(m_config.cell_count() * 2);
      for(size_t idx = 0; idx < m_config.cell_count(); ++idx) {
         auto first = make_cell(idx);
         // canonical orientation (right neighbour, down neighbour) to avoid duplicates
         if(size_t(first.col) + 1 < m_config.cols) {
            out.emplace_back(Place{first, Cell{first.row, int8_t(first.col + 1)}});
         }
         if(size_t(first.row) + 1 < m_config.rows) {
            out.emplace_back(Place{first, Cell{int8_t(first.row + 1), first.col}});
         }
      }
      std::erase_if(out, [&](const Action& action) { return not is_valid(action); });
      return out;
   }
   out.reserve(m_config.cell_count());
   for(size_t idx = 0; idx < m_config.cell_count(); ++idx) {
      out.emplace_back(Fire{make_cell(idx)});
   }
   return out;
}

inline bool State::fleet_sunk(Player player) const
{
   return _all_ships_sunk_of(player);
}

inline bool State::ship_sunk(Player player, size_t ship_index) const
{
   const size_t base = 2 * ship_index;
   const auto& fleet = m_fleets[as_int(player)];
   if(base + 1 >= fleet.size()) {
      throw std::out_of_range("battleship: no such ship in the fleet of the given player.");
   }
   const auto& foe_hits = m_hits[as_int(opponent(player))];
   return foe_hits[cell_index(fleet[base])] and foe_hits[cell_index(fleet[base + 1])];
}

inline bool State::_overlaps_own_fleet(Player placer, Cell a, Cell b) const
{
   const auto& fleet = m_fleets[as_int(placer)];
   for(const auto& cell : fleet) {
      if(cell == a or cell == b) {
         return true;
      }
   }
   return false;
}

inline bool State::_all_ships_sunk_of(Player fleet_owner) const
{
   const auto& fleet = m_fleets[as_int(fleet_owner)];
   const auto& foe_hits = m_hits[as_int(opponent(fleet_owner))];
   if(fleet.size() < 2 * m_config.ships_per_fleet) {
      // the fleet is not fully placed yet
      return false;
   }
   for(size_t i = 0; i + 1 < fleet.size(); i += 2) {
      if(not foe_hits[cell_index(fleet[i])] or not foe_hits[cell_index(fleet[i + 1])]) {
         return false;
      }
   }
   return true;
}

}  // namespace battleship

#endif  // NOR_BATTLESHIP_STATE_HPP
