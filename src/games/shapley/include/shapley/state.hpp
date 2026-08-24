
#ifndef NOR_SHAPLEY_STATE_HPP
#define NOR_SHAPLEY_STATE_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace shapley {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent (cf. pursuit_evasion /
/// oshi_zumo).
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
// SOURCE. Shapley, L. S. (1964), "Some Topics in Two-Person Games", in: Dresher/Shapley/Tucker
// (eds.), Advances in Game Theory, Annals of Mathematics Studies 52, Princeton University Press,
// pp. 1-28 -- the section 5.3 counterexample. This is THE canonical 3x3 general-sum game whose
// fictitious-play best-response dynamics cycle forever without converging to the (unique,
// interior) Nash equilibrium; Shapley originally defined the game through its best-response
// ordering, and the following numeric bimatrix is the standard faithful realization of that
// ordering (transcribed verbatim from the public-domain fictitious-play simulation of Shapley
// section 5.3, J. Bhattacharya, gist.github.com/jmoy/8541995; ordinal-equivalent rescalings of
// the identical preference graph appear in Krishna/Sjostrom 1998 "On the Convergence of
// Fictitious Play" and Gaunersdorfer/Hofbauer 1995 as [[0,0],[2,1],[1,2]]-type matrices).
//
// BIMATRIX (rows = player one's strategies top/middle/bottom, cols = player two's left/middle/
// right; entry = (u_one, u_two)):
//
//            left      middle    right
//   top     ( 1, 0)   ( 0, 0)   ( 0, 1)
//   middle  ( 0, 1)   ( 1, 0)   ( 0, 0)
//   bottom  ( 0, 0)   ( 0, 1)   ( 1, 0)
//
// STRUCTURAL PROPERTIES (all asserted in the unit tests straight off these matrices):
// - GENERAL-SUM: e.g. (top, left) yields (1, 0) with nonzero payoff sum.
// - Pure best responses form the famous best-response cycle. Read off column-wise /
//   row-wise:
//       BR_1(left)={top},    BR_1(middle)={middle}, BR_1(right)={bottom},
//       BR_2(top)={right},   BR_2(middle)={left},   BR_2(bottom)={middle},
//   so cycling fictitious play visits the six profiles
//       (top,left) -> (top,right) -> (bottom,right) -> (bottom,middle)
//                  -> (middle,middle) -> (middle,left) -> back,
//   and never the antidiagonal {(top,middle), (middle,right), (bottom,left)}.
// - Unique Nash equilibrium: both players uniform (1/3, 1/3, 1/3); equilibrium payoffs (1/3,1/3).
//
// SEQUENTIALIZATION. The simultaneous normal-form move is sequentialized as
// CommitP1 -> CommitP2(-fused-resolve) exactly like the deterministic oshi_zumo encoding: the
// resolve is a deterministic transition fused into the application of player two's commitment.
// Neither commitment reveals its value, so player two's infoset spans all three of his commit
// histories and the extensive-form game realizes exactly the simultaneous bimatrix above
// (commit-commit with hidden values <=> simultaneous move selection).

/// row/column strategy index in [0, 3): 0=top|left, 1=middle, 2=bottom|right
struct Play {
   uint8_t strategy = 0;

   friend bool operator==(const Play&, const Play&) = default;
   friend bool operator!=(const Play&, const Play&) = default;
};

/// number of pure strategies per player (fixed by the citation; nothing is parameterized)
constexpr size_t strategy_count = 3;

/// player one's payoff matrix (row major, [row][col])
inline constexpr std::array< std::array< double, strategy_count >, strategy_count > k_payoff_one{
   {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};

/// player two's payoff matrix (row major, [row][col])
inline constexpr std::array< std::array< double, strategy_count >, strategy_count > k_payoff_two{
   {{{0., 0., 1.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};

/**
 * @brief The phase of the sequentialized joint move.
 *
 * CommitP1 -> CommitP2 with the deterministic resolve fused into the CommitP2 application (repo
 * convention for deterministic environments, cf. oshi_zumo): applying the second Play applies
 * both plays simultaneously at once, keeping every non-terminal state attributable to a real
 * acting player (which deterministic-env CFR traversal requires).
 */
enum class Phase : uint8_t { commit_p1 = 0, commit_p2 = 1 };

/**
 * @brief World state of one Shapley-game instance.
 *
 * The whole game is a single sequentialized joint move; all data is fixed-size and copying stays
 * trivially cheap.
 */
class State {
  public:
   State() = default;

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   /**
    * Commits a play for whoever is due. During CommitP1 nothing but the commitment is stored;
    * the joint resolution happens atomically inside the CommitP2 application (fused
    * deterministic resolve, see Phase).
    */
   void apply_action(Play play)
   {
      if(terminal()) {
         throw std::logic_error("shapley: state is terminal; no further plays.");
      }
      switch(m_phase) {
         case Phase::commit_p1: {
            if(not is_valid(Player::one, play)) {
               throw std::invalid_argument(
                  "shapley: illegal player-one strategy index " + std::to_string(play.strategy)
                  + "."
               );
            }
            m_committed.at(as_int(Player::one)) = play;
            m_phase = Phase::commit_p2;
            m_active = Player::two;
            return;
         }
         case Phase::commit_p2: {
            if(not is_valid(Player::two, play)) {
               throw std::invalid_argument(
                  "shapley: illegal player-two strategy index " + std::to_string(play.strategy)
                  + "."
               );
            }
            m_committed.at(as_int(Player::two)) = play;
            m_active = Player::none;  // fused resolve terminates the game
            return;
         }
      }
      throw std::logic_error("shapley: unreachable phase.");
   }

   [[nodiscard]] bool is_valid(Play play) const
   {
      if(terminal()) {
         return false;
      }
      switch(m_phase) {
         case Phase::commit_p1: return is_valid(Player::one, play);
         case Phase::commit_p2: return is_valid(Player::two, play);
      }
      return false;
   }

   /// legality of `play` for `player` independent of the current phase
   [[nodiscard]] static bool is_valid(Player player, Play play)
   {
      switch(player) {
         case Player::one:
         case Player::two: return play.strategy < strategy_count;
         default: return false;
      }
   }

   /// legal plays of whoever is due to commit (empty outside the commit phases / when terminal)
   [[nodiscard]] std::vector< Play > actions(Player player) const
   {
      std::vector< Play > out;
      if(terminal() or m_active != player) {
         return out;
      }
      out.reserve(strategy_count);
      for(uint8_t s = 0; s < strategy_count; ++s) {
         out.push_back(Play{s});
      }
      return out;
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_committed.at(as_int(Player::two)).has_value(); }

   /**
    * @brief GENERAL-SUM terminal payoffs transcribed from the canonical bimatrix.
    *
    * u(one) = A[row][col] and u(two) = B[row][col] with the matrices above; 0 before terminality.
    * The payoffs deliberately do NOT sum to zero in general.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal()) {
         return 0.;
      }
      const auto row = m_committed.at(as_int(Player::one))->strategy;
      const auto col = m_committed.at(as_int(Player::two))->strategy;
      switch(player) {
         case Player::one: return k_payoff_one.at(row).at(col);
         case Player::two: return k_payoff_two.at(row).at(col);
         default: return 0.;
      }
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   /// the pure best responses of `player` against a fixed opponent strategy (matrix lookup);
   /// exposed for the best-response-cycle truth-table tests
   [[nodiscard]] static std::vector< uint8_t > best_responses(Player player, uint8_t opp_strategy)
   {
      const auto& matrix = player == Player::one ? k_payoff_one : k_payoff_two;
      double best = -1.;
      std::vector< uint8_t > out;
      for(size_t s = 0; s < strategy_count; ++s) {
         const double value = player == Player::one ? matrix.at(s).at(opp_strategy)
                                                    : matrix.at(opp_strategy).at(s);
         if(value > best) {
            best = value;
            out.clear();
            out.push_back(uint8_t(s));
         } else if(value == best) {
            out.push_back(uint8_t(s));
         }
      }
      return out;
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] Phase phase() const { return m_phase; }
   /// nullopt while the respective side has not committed yet
   [[nodiscard]] std::optional< Play > committed_play(Player player) const
   {
      return m_committed.at(as_int(player));
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_committed == other.m_committed && m_phase == other.m_phase
             && m_active == other.m_active;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   /////////////////////////////////
   /// API: data members        ///
   /////////////////////////////////

   std::array< std::optional< Play >, 2 > m_committed{};
   Phase m_phase = Phase::commit_p1;
   Player m_active = Player::one;
};

}  // namespace shapley

#endif  // NOR_SHAPLEY_STATE_HPP
