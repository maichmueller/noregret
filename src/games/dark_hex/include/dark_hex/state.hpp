
#ifndef NOR_DARK_HEX_STATE_HPP
#define NOR_DARK_HEX_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dark_hex {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent
enum class Player : int8_t { none = -2, one = 0, two = 1 };

[[nodiscard]] constexpr Player opponent(Player player)
{
   return player == Player::one ? Player::two : Player::one;
}

/**
 * @brief The rules variant governing failed attempts at occupied cells.
 *
 * In both variants an attempt at a cell that is already occupied by EITHER player fails and
 * places no stone. The variants differ in the referee feedback:
 * - cdh ("classical dark hex", OpenSpiel default): the actor is TOLD that his attempt failed and
 *   RETAINS the turn (he may retry immediately).
 * - adh ("abrupt dark hex"): the failure is silent to gameplay; the turn simply passes on.
 */
enum class RulesMode : uint8_t { cdh = 0, adh };

template < typename To = size_t, typename T >
[[nodiscard]] constexpr To as_int(T value)
{
   return static_cast< To >(value);
}

/// the maximal number of cells of the board (n <= 5, flattened row-major)
constexpr size_t max_cells = 25;

/**
 * @brief A single attempt to place a stone on the flattened board cell 'cell_index'.
 *
 * Every in-grid cell is always an available action: attempting an occupied cell is legal but
 * resolves as a failure (no stone placed) instead of being rejected upfront -- this is what makes
 * the game one of imperfect information.
 */
struct Move {
   uint8_t cell_index = 0;

   friend bool operator==(const Move&, const Move&) = default;
   friend bool operator!=(const Move&, const Move&) = default;
};

/**
 * @brief Configuration of a dark hex instance.
 *
 * @param board_size side length n of the n x n rhombus; supported range 2..5 (the published
 * benchmark sizes are 3..5, with 3 the default for tests; 2 remains constructible for cheap CFR
 * smoke tests).
 * @param rules_mode how failed attempts are handled (see RulesMode).
 */
struct Config {
   size_t board_size = 3;
   RulesMode rules_mode = RulesMode::cdh;

   Config() = default;
   Config(size_t board_size_, RulesMode rules_mode_)
       : board_size(board_size_),
         rules_mode(rules_mode_)
   {
      validate();
   }

   [[nodiscard]] size_t cell_count() const { return board_size * board_size; }

   void validate() const
   {
      if(board_size < 2 or board_size > 5) {
         throw std::invalid_argument(
            "dark hex supports boards of size 2..5 (uint32 bitmask + uint8 cell indices)."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/**
 * @brief One chronological entry of the world state's move log.
 *
 * The full log is part of the state because failed attempts leave no other trace: without it the
 * environment could not reconstruct which observations the players received (a player privately
 * observes each of his own attempts, successful or not).
 */
struct LogEntry {
   Player actor = Player::none;
   uint8_t cell_index = 0;

   friend bool operator==(const LogEntry&, const LogEntry&) = default;
   friend bool operator!=(const LogEntry&, const LogEntry&) = default;
};

/**
 * @brief The world state of a single deterministic dark hex game.
 *
 * Imperfect information arises purely from unobserved opponent moves plus referee feedback:
 * every player sees only his own stones. All data lives in fixed-size members or small vectors
 * so copying (which happens for every edge of the search tree) stays cheap.
 */
class State {
   public:
   explicit State(Config config = {}) : m_config(config) { m_config.validate(); }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   void apply_action(const Move& move)
   {
      if(terminal()) {
         throw std::logic_error("dark hex state is terminal; no further actions can be applied.");
      }
      if(not is_valid(move)) {
         throw std::invalid_argument("illegal dark hex move (cell index out of grid).");
      }
      const auto actor = m_active;
      const auto bit = uint32_t{1} << move.cell_index;
      if(((m_stones[0] | m_stones[1]) & bit) != 0u) {
         // the attempted cell is occupied by either player: the attempt FAILS silently in the
         // world (no stone), but is logged for observation purposes.
         m_last_attempt_failed = true;
         if(m_config.rules_mode == RulesMode::adh) {
            // abrupt mode: a consumed turn passes on without further comment
            m_active = opponent(actor);
         }
         // classical mode: the referee tells the actor about the rejection and he retains the turn
      } else {
         m_last_attempt_failed = false;
         m_stones[as_int(actor)] |= bit;
         if(has_won(actor)) {
            m_active = Player::none;
         } else {
            m_active = opponent(actor);
         }
      }
      m_attempts[as_int(actor)] += 1;
      m_move_count += 1;
      m_log.push_back(LogEntry{actor, move.cell_index});
   }

   [[nodiscard]] bool is_valid(const Move& move) const
   {
      return move.cell_index < m_config.cell_count();
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_active == Player::none; }

   /**
    * zero-sum terminal payoff: the connecting player receives +1, his opponent -1. Non-terminal
    * states score 0 for everyone (draws are impossible in hex once the game runs out).
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal()) {
         return 0.;
      }
      return has_won(player) ? 1. : -1.;
   }

   /**
    * every in-grid cell is a legal action candidate for the active player (occupied targets fail
    * upon application rather than being filtered here)
    */
   [[nodiscard]] std::vector< Move > actions(Player player) const
   {
      std::vector< Move > out;
      if(terminal() or active_player() != player) {
         return out;
      }
      out.reserve(m_config.cell_count());
      for(size_t idx = 0; idx < m_config.cell_count(); ++idx) {
         out.push_back(Move{uint8_t(idx)});
      }
      return out;
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] const Config& config() const { return m_config; }
   [[nodiscard]] size_t move_count() const { return m_move_count; }
   /// whether the most recent attempt was rejected (occupied target); false after placements
   [[nodiscard]] bool last_attempt_failed() const { return m_last_attempt_failed; }
   /// number of attempts (successful or not) made by `player`
   [[nodiscard]] size_t attempts(Player player) const { return m_attempts[as_int(player)]; }
   /// the chronological record of all attempts, in application order
   [[nodiscard]] const std::vector< LogEntry >& move_log() const { return m_log; }

   /// the stones of `player` as a bitmask over the flattened board
   [[nodiscard]] uint32_t stones(Player player) const { return m_stones[as_int(player)]; }
   /// the cells occupied by either player
   [[nodiscard]] uint32_t occupancy() const { return m_stones[0] | m_stones[1]; }
   [[nodiscard]] bool is_occupied(size_t cell_index) const
   {
      return ((m_stones[0] | m_stones[1]) >> cell_index) & uint32_t{1};
   }

   /// whether `player` has already completed his winning connection
   [[nodiscard]] bool has_won(Player player) const
   {
      // player one connects top <-> bottom (row 0 to row n-1), player two left <-> right
      return connected(m_stones[as_int(player)], player == Player::one);
   }

   /**
    * hex connectivity via iterative flood fill over the bitmask: seeds one board edge owned by
    * `mask`'s owner and expands along hex adjacency until the opposite edge is reached.
    * `top_bottom` selects which pair of edges to bridge (rows for player one, columns otherwise).
    */
   [[nodiscard]] bool connected(uint32_t mask, bool top_bottom) const
   {
      if(mask == 0) {
         return false;
      }
      const size_t n = m_config.board_size;
      uint32_t visited = 0;
      std::array< uint8_t, max_cells > stack{};
      size_t sp = 0;
      for(size_t i = 0; i < n; ++i) {
         const size_t idx = top_bottom ? i : i * n;
         if(((mask >> idx) & uint32_t{1}) != 0u) {
            visited |= uint32_t{1} << idx;
            stack[sp++] = uint8_t(idx);
         }
      }
      const size_t last = n - 1;
      while(sp > 0) {
         const size_t idx = stack[--sp];
         const size_t row = idx / n;
         const size_t col = idx % n;
         if((top_bottom ? row : col) == last) {
            return true;
         }
         // neighbors of (r,c): (r,c-1),(r,c+1),(r-1,c),(r+1,c),(r-1,c+1),(r+1,c-1)
         constexpr int8_t drow[6] = {0, 0, -1, 1, -1, 1};
         constexpr int8_t dcol[6] = {-1, 1, 0, 0, 1, -1};
         for(size_t d = 0; d < 6; ++d) {
            const auto nr = int(row) + drow[d];
            const auto nc = int(col) + dcol[d];
            if(nr < 0 or nc < 0 or size_t(nr) >= n or size_t(nc) >= n) {
               continue;
            }
            const size_t nidx = size_t(nr) * n + size_t(nc);
            if((((mask >> nidx) & uint32_t{1}) != 0u) and ((visited >> nidx) & uint32_t{1}) == 0u) {
               visited |= uint32_t{1} << nidx;
               stack[sp++] = uint8_t(nidx);
            }
         }
      }
      return false;
   }

   [[nodiscard]] size_t cell_index(size_t row, size_t col) const
   {
      return row * m_config.board_size + col;
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_active == other.m_active
             && m_stones[0] == other.m_stones[0] && m_stones[1] == other.m_stones[1]
             && m_last_attempt_failed == other.m_last_attempt_failed
             && m_move_count == other.m_move_count && m_log == other.m_log;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not(*this == other); }

   private:
   Config m_config{};
   Player m_active = Player::one;
   std::array< uint32_t, 2 > m_stones{0u, 0u};
   bool m_last_attempt_failed = false;
   size_t m_move_count = 0;
   std::array< size_t, 2 > m_attempts{0, 0};
   std::vector< LogEntry > m_log{};
};

}  // namespace dark_hex

#endif  // NOR_DARK_HEX_STATE_HPP
